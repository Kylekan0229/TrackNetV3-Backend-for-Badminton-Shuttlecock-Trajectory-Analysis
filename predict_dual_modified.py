import os
import json
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """Predict coordinates from heatmap or inpainted coordinates."""
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()

    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)

    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    c_p = c_pred[n][f]
                    cx_pred = int(c_p[0] * WIDTH * img_scaler[0])
                    cy_pred = int(c_p[1] * HEIGHT * img_scaler[1])
                elif y_pred is not None:
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred = int(bbox_pred[0] + bbox_pred[2] / 2)
                    cy_pred = int(bbox_pred[1] + bbox_pred[3] / 2)
                    cx_pred = int(cx_pred * img_scaler[0])
                    cy_pred = int(cy_pred * img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break

    return pred_dict


def _select_two_videos_with_dialog() -> Tuple[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError(
            'Cannot open file picker. Please install tkinter or pass --video_file_a and --video_file_b directly.'
        ) from e

    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title='Select exactly 2 video files',
        filetypes=[('Video files', '*.mp4 *.mov *.m4v *.avi'), ('All files', '*.*')]
    )
    root.update()
    root.destroy()

    if len(file_paths) != 2:
        raise ValueError(f'Please select exactly 2 videos. You selected {len(file_paths)} file(s).')
    return file_paths[0], file_paths[1]


def _load_models(args, device):
    tracknet_ckpt = torch.load(args.tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])

    inpaintnet = None
    inpaintnet_seq_len = None
    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, map_location=device)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])

    return tracknet, tracknet_seq_len, bg_mode, inpaintnet, inpaintnet_seq_len


def _run_single_video(video_file: str, args, device, tracknet, tracknet_seq_len, bg_mode,
                      inpaintnet=None, inpaintnet_seq_len=None):
    num_workers = 0

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_file}')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    cap.release()

    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {
        'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)
    }

    tracknet.eval()
    seq_len = tracknet_seq_len

    if args.eval_mode == 'nonoverlap':
        if large_video:
            dataset = Video_IterableDataset(
                video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                max_sample_num=args.max_sample_num, video_range=video_range
            )
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            print(f'[{video_name}] Video length: {dataset.video_len}')
        else:
            frame_list = generate_frames(video_file)
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True
            )
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for _, (i, x) in enumerate(tqdm(data_loader, desc=f'TrackNet {video_name}')):
            x = x.float().to(device)
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

    else:
        if large_video:
            dataset = Video_IterableDataset(
                video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                max_sample_num=args.max_sample_num, video_range=video_range
            )
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'[{video_name}] Video length: {video_len}')
        else:
            frame_list = generate_frames(video_file)
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                frame_arr=np.array(frame_list)[:, :, :, ::-1]
            )
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            video_len = len(frame_list)

        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, args.eval_mode)

        for _, (i, x) in enumerate(tqdm(data_loader, desc=f'TrackNet {video_name}')):
            x = x.float().to(device)
            b_size = i.shape[0]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    y_e = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    y_e = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_e.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        y_last = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_last.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # optional InpaintNet
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

        if args.eval_mode == 'nonoverlap':
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True
            )
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for _, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader, desc=f'InpaintNet {video_name}')):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
        else:
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)
            frame_i = torch.arange(seq_len - 1, -1, -1)
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

            for _, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader, desc=f'InpaintNet {video_name}')):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        c_e = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        c_e = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, c_e.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        for f in range(1, seq_len):
                            c_last = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, c_last.view(1, 1, 2)), dim=0)

                th_mask2 = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask2] = 0.
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])

                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        pred_dict = inpaint_pred_dict
    else:
        pred_dict = tracknet_pred_dict

    meta = {
        'video_file': os.path.abspath(video_file),
        'video_name': video_name,
        'video_width': w,
        'video_height': h,
        'fps': fps,
        'device': str(device),
        'tracknet_input_width': WIDTH,
        'tracknet_input_height': HEIGHT,
        'img_scaler': [w_scaler, h_scaler],
        'csv_coord_space': 'video_pixels',
        'csv_columns': ['Frame', 'Visibility', 'X', 'Y'],
    }

    return pred_dict, meta


def _pred_to_dataframe(pred_dict: dict, fps: float, suffix: str) -> pd.DataFrame:
    df = pd.DataFrame({
        f'Frame_{suffix}': pred_dict['Frame'],
        f'Visibility_{suffix}': pred_dict['Visibility'],
        f'X_{suffix}': pred_dict['X'],
        f'Y_{suffix}': pred_dict['Y'],
    })
    df[f'TimeSec_{suffix}'] = df[f'Frame_{suffix}'] / float(fps)
    return df


def _combine_predictions(pred_a: dict, meta_a: dict, pred_b: dict, meta_b: dict) -> pd.DataFrame:
    df_a = _pred_to_dataframe(pred_a, meta_a['fps'], 'A').sort_values('TimeSec_A').reset_index(drop=True)
    df_b = _pred_to_dataframe(pred_b, meta_b['fps'], 'B').sort_values('TimeSec_B').reset_index(drop=True)

    tolerance = max(1.0 / max(meta_a['fps'], 1.0), 1.0 / max(meta_b['fps'], 1.0))
    merged = pd.merge_asof(
        df_a,
        df_b,
        left_on='TimeSec_A',
        right_on='TimeSec_B',
        direction='nearest',
        tolerance=tolerance,
    )
    merged.insert(0, 'StereoIndex', np.arange(len(merged), dtype=int))
    merged.insert(1, 'TimeSec', merged['TimeSec_A'])
    merged['TimeDeltaMs'] = (merged['TimeSec_B'] - merged['TimeSec_A']) * 1000.0
    return merged


def _default_prefix(video_a: str, video_b: str) -> str:
    name_a = os.path.splitext(os.path.basename(video_a))[0]
    name_b = os.path.splitext(os.path.basename(video_b))[0]
    return f'{name_a}__{name_b}'


def _write_meta(save_path: str, content: dict):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run TrackNetV3 on 2 videos and save both results into one combined CSV.')
    parser.add_argument('--video_file_a', type=str, default='', help='path of the first video')
    parser.add_argument('--video_file_b', type=str, default='', help='path of the second video')
    parser.add_argument('--pick_videos', action='store_true', help='open a file picker to select exactly 2 videos')
    parser.add_argument('--tracknet_file', type=str, required=True, help='path of the TrackNet checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='path of the InpaintNet checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='start_second,end_second for median image')
    parser.add_argument('--save_dir', type=str, default='pred_result_dual', help='directory to save outputs')
    parser.add_argument('--save_prefix', type=str, default='', help='prefix for the combined stereo CSV / meta files')
    parser.add_argument('--large_video', action='store_true', default=False, help='process large video with iterable dataset')
    parser.add_argument('--output_video', action='store_true', default=False, help='also output 2 videos with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='trajectory tail length')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='inference device')
    args = parser.parse_args()

    if args.pick_videos or (not args.video_file_a and not args.video_file_b):
        args.video_file_a, args.video_file_b = _select_two_videos_with_dialog()
    elif not args.video_file_a or not args.video_file_b:
        raise ValueError('Please provide both --video_file_a and --video_file_b, or use --pick_videos.')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    tracknet, tracknet_seq_len, bg_mode, inpaintnet, inpaintnet_seq_len = _load_models(args, device)

    pred_a, meta_a = _run_single_video(args.video_file_a, args, device, tracknet, tracknet_seq_len, bg_mode, inpaintnet, inpaintnet_seq_len)
    pred_b, meta_b = _run_single_video(args.video_file_b, args, device, tracknet, tracknet_seq_len, bg_mode, inpaintnet, inpaintnet_seq_len)

    base_a = os.path.splitext(os.path.basename(args.video_file_a))[0]
    base_b = os.path.splitext(os.path.basename(args.video_file_b))[0]
    prefix = args.save_prefix.strip() or _default_prefix(args.video_file_a, args.video_file_b)

    csv_a_path = os.path.join(args.save_dir, f'{base_a}_ball.csv')
    csv_b_path = os.path.join(args.save_dir, f'{base_b}_ball.csv')
    write_pred_csv(pred_a, csv_a_path)
    write_pred_csv(pred_b, csv_b_path)

    meta_a['csv_path'] = os.path.abspath(csv_a_path)
    meta_b['csv_path'] = os.path.abspath(csv_b_path)
    _write_meta(os.path.join(args.save_dir, f'{base_a}_meta.json'), meta_a)
    _write_meta(os.path.join(args.save_dir, f'{base_b}_meta.json'), meta_b)

    combined_df = _combine_predictions(pred_a, meta_a, pred_b, meta_b)
    combined_csv_path = os.path.join(args.save_dir, f'{prefix}_stereo.csv')
    combined_df.to_csv(combined_csv_path, index=False)

    stereo_meta = {
        'video_file_a': os.path.abspath(args.video_file_a),
        'video_file_b': os.path.abspath(args.video_file_b),
        'video_name_a': base_a,
        'video_name_b': base_b,
        'fps_a': meta_a['fps'],
        'fps_b': meta_b['fps'],
        'video_width_a': meta_a['video_width'],
        'video_height_a': meta_a['video_height'],
        'video_width_b': meta_b['video_width'],
        'video_height_b': meta_b['video_height'],
        'csv_a_path': os.path.abspath(csv_a_path),
        'csv_b_path': os.path.abspath(csv_b_path),
        'combined_csv_path': os.path.abspath(combined_csv_path),
        'combined_csv_columns': list(combined_df.columns),
        'combine_method': 'merge_asof on timestamps with nearest-frame tolerance',
        'note': (
            'This combined CSV keeps Camera A and Camera B detections on the same row using timestamp-based alignment. '
            'It is intended for the next stereo / triangulation stage, not as a final 3D result by itself.'
        )
    }
    _write_meta(os.path.join(args.save_dir, f'{prefix}_stereo_meta.json'), stereo_meta)

    if args.output_video:
        write_pred_video(args.video_file_a, pred_a, save_file=os.path.join(args.save_dir, f'{base_a}_pred.mp4'), traj_len=args.traj_len)
        write_pred_video(args.video_file_b, pred_b, save_file=os.path.join(args.save_dir, f'{base_b}_pred.mp4'), traj_len=args.traj_len)

    print('\nDone.')
    print(f'Camera A CSV : {csv_a_path}')
    print(f'Camera B CSV : {csv_b_path}')
    print(f'Combined CSV : {combined_csv_path}')


if __name__ == '__main__':
    main()
