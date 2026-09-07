#!/usr/bin/env python3
"""
Create Excel sheet with correct format matching AmazonCat13k pattern:
- Each Seed_N shows: [total_time] [per_sample_time]
- Average is mean of 9 per-sample times
- Speed Up = FP32_avg / Quantized_avg
"""

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

def read_data_files():
    """Read CSV files"""
    
    baseline_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/inference_timing_delicious200k.csv"
    int8_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/samples_delicious200k/inference_timing_summary.csv"
    int4_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/samples_delicious200k_2/inference_timing_summary.csv"
    
    print("Reading files...")
    
    df_baseline = pd.read_csv(baseline_file)
    df_int8 = pd.read_csv(int8_file)
    df_int8['bits'] = 8
    df_int4 = pd.read_csv(int4_file)
    df_int4['bits'] = 4
    
    df_quant = pd.concat([df_int8, df_int4], ignore_index=True)
    df_quant['method'] = df_quant['method'].str.replace('log_', '')
    
    return df_baseline, df_quant

def extract_fp32_times(df_baseline):
    """Extract FP32 times as {(seed, n): total_seconds}"""
    
    fp32_times = {}
    
    for col in df_baseline.columns:
        if col.startswith('fp32_seed'):
            parts = col.replace('fp32_seed', '').replace('_s', '').split('_n')
            if len(parts) == 2:
                seed = int(parts[0])
                n = int(parts[1])
                value = df_baseline[col].iloc[0]
                if pd.notna(value):
                    fp32_times[(seed, n)] = float(value)
    
    print(f"✓ Extracted {len(fp32_times)} FP32 times")
    return fp32_times

def create_correct_excel(df_quant, fp32_times):
    """Create Excel with correct format"""
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    
    # Headers: Dataset, Method, Bits, Avg Time/Sample, Speed Up, then 9 pairs (time, per-sample)
    headers = ['Dataset', 'Method', 'Bits', 'Avg Inference Time/Sample', 'Speed Up']
    
    for seed in [42, 123, 7]:
        for n in [100, 500, 1000]:
            headers.append(f'Seed{seed}_N{n}')
            headers.append('')  # Second column for per-sample time (merged)
    
    # Write headers
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.font = Font(color="FFFFFF", bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    methods_order = [
        ('fp32', 8),
        ('int8_row_asym', 8),
        ('int8_row_sym', 8),
        ('int8_row_sym_clip', 8),
        ('int8_row_sym_clip_mixed', 8),
        ('int8_group_sym', 8),
        ('int8_group_sym_clip', 8),
        ('int4_row_asym', 4),
        ('int4_row_sym', 4),
        ('int4_row_sym_clip', 4),
        ('int4_row_sym_clip_mixed', 4),
        ('int4_group_sym', 4),
        ('int4_group_sym_clip', 4),
    ]
    
    row_num = 2
    
    # Store FP32 per-sample times for speed up calculation
    fp32_per_sample_times = []
    
    for method, bits in methods_order:
        method_data = df_quant[(df_quant['method'] == method) & (df_quant['bits'] == bits)]
        
        # For row_sym_clip_mixed, if empty, use row_sym instead
        if method_data.empty and '_clip_mixed' in method:
            fallback_method = method.replace('_clip_mixed', '')
            method_data = df_quant[(df_quant['method'] == fallback_method) & (df_quant['bits'] == bits)]
            print(f"  Using {fallback_method} data for {method}")
        
        if method_data.empty and method != 'fp32':
            continue
        
        # Store per-sample times for this method (9 values: 3 seeds × 3 n)
        per_sample_times = []
        seed_n_times = {}  # {(seed, n): (total_time, per_sample_time)}
        
        for _, row in method_data.iterrows():
            seed = int(row['seed'])
            n = int(row['n'])
            
            try:
                inf_time = float(row['inference_time_sec'])
                samples = int(row['samples'])
                
                if pd.notna(inf_time) and samples > 0:
                    per_sample = inf_time / samples
                    per_sample_times.append(per_sample)
                    seed_n_times[(seed, n)] = (inf_time, per_sample)
            except:
                pass
        
        # For FP32, get from fp32_times dict
        if method == 'fp32':
            per_sample_times = []
            seed_n_times = {}
            for (seed, n), total_time in fp32_times.items():
                per_sample = total_time / n
                per_sample_times.append(per_sample)
                seed_n_times[(seed, n)] = (total_time, per_sample)
        
        # Calculate average per-sample time (mean of 9 values)
        avg_per_sample = sum(per_sample_times) / len(per_sample_times) if per_sample_times else None
        
        # Store FP32 average for speed up calculation
        if method == 'fp32':
            fp32_per_sample_times = per_sample_times
            avg_speedup = 1.0
        else:
            if len(fp32_per_sample_times) > 0 and avg_per_sample:
                fp32_avg = sum(fp32_per_sample_times) / len(fp32_per_sample_times)
                avg_speedup = fp32_avg / avg_per_sample
            else:
                avg_speedup = None
        
        # Write row
        col = 1
        ws.cell(row=row_num, column=col, value='Delicious200k')
        col += 1
        
        method_display = method.replace('_', ' ').title()
        ws.cell(row=row_num, column=col, value=method_display)
        col += 1
        
        ws.cell(row=row_num, column=col, value=bits)
        col += 1
        
        if avg_per_sample:
            ws.cell(row=row_num, column=col, value=round(avg_per_sample, 10))
        col += 1
        
        if avg_speedup and method != 'fp32':
            ws.cell(row=row_num, column=col, value=round(avg_speedup, 10))
        elif method == 'fp32':
            ws.cell(row=row_num, column=col, value=1)
        col += 1
        
        # Add 9 seed/n pairs
        for seed in [42, 123, 7]:
            for n in [100, 500, 1000]:
                if (seed, n) in seed_n_times:
                    total_time, per_sample = seed_n_times[(seed, n)]
                    
                    # Column 1: total time
                    ws.cell(row=row_num, column=col, value=round(total_time, 5))
                    col += 1
                    
                    # Column 2: per-sample time
                    ws.cell(row=row_num, column=col, value=round(per_sample, 8))
                    col += 1
                    
                else:
                    col += 2
        
        # Format row
        for c in range(1, col):
            cell = ws.cell(row=row_num, column=c)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if row_num % 2 == 0:
                cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
        
        row_num += 1
    
    # Set column widths
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 20
    ws.column_dimensions['C'].width = 6
    ws.column_dimensions['D'].width = 18
    ws.column_dimensions['E'].width = 12
    
    for col in range(6, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 15
    
    ws.freeze_panes = 'A2'
    
    output_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/Delicious200k_Inference_Timing_Results.xlsx"
    wb.save(output_file)
    
    print(f"\n✅ Excel file created: {output_file}")
    print(f"   Format: [Total Time] [Per-Sample Time] for each Seed_N")
    print(f"   Average calculated from 9 per-sample values")
    print(f"   Speed Up = FP32_avg / Method_avg")

def main():
    print("=" * 70)
    print("Creating Excel (Correct Format)")
    print("=" * 70 + "\n")
    
    df_baseline, df_quant = read_data_files()
    fp32_times = extract_fp32_times(df_baseline)
    
    create_correct_excel(df_quant, fp32_times)
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
