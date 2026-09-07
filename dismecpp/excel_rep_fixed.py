#!/usr/bin/env python3
"""
Create Excel sheet with correct FP32 baseline from inference_timing_delicious200k.csv
"""

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def read_data_files():
    """Read both CSV files and extract baseline and quantized data"""
    
    # File with FP32 baseline
    baseline_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/inference_timing_delicious200k.csv"
    
    # Files with quantized results
    int8_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/samples_delicious200k/inference_timing_summary.csv"
    int4_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/samples_delicious200k_2/inference_timing_summary.csv"
    
    print("Reading data files...")
    
    # Read baseline FP32 times
    df_baseline = pd.read_csv(baseline_file)
    print(f"✓ Baseline file: {len(df_baseline)} rows")
    print(f"  Columns: {list(df_baseline.columns)[:10]}...")
    
    # Read INT8 data
    df_int8 = pd.read_csv(int8_file)
    df_int8['bits'] = 8
    print(f"✓ INT8 data: {len(df_int8)} rows")
    
    # Read INT4 data
    df_int4 = pd.read_csv(int4_file)
    df_int4['bits'] = 4
    print(f"✓ INT4 data: {len(df_int4)} rows")
    
    # Combine quantized data
    df_quant = pd.concat([df_int8, df_int4], ignore_index=True)
    df_quant['method'] = df_quant['method'].str.replace('log_', '')
    
    return df_baseline, df_quant

def extract_fp32_baselines(df_baseline):
    """Extract FP32 baseline times from the baseline file"""
    
    baselines = {}
    
    # Extract from columns like fp32_seed42_n100_s, fp32_seed42_n500_s, etc.
    for col in df_baseline.columns:
        if col.startswith('fp32_seed'):
            # Parse column name: fp32_seed42_n100_s -> (42, 100)
            parts = col.replace('fp32_seed', '').replace('_s', '').split('_n')
            if len(parts) == 2:
                seed = int(parts[0])
                n = int(parts[1])
                
                # Get value from first row (all rows have same FP32 times)
                value = df_baseline[col].iloc[0]
                if pd.notna(value):
                    baselines[(seed, n)] = float(value)
    
    print(f"\n✓ Extracted {len(baselines)} FP32 baseline times:")
    for (seed, n), time_val in sorted(baselines.items()):
        print(f"  Seed{seed}_N{n}: {time_val:.6f} sec")
    
    return baselines

def calculate_per_sample_time(row):
    """Calculate inference time per sample"""
    try:
        inf_time = float(row['inference_time_sec'])
        samples = float(row['samples'])
        if pd.notna(inf_time) and pd.notna(samples) and samples > 0:
            return inf_time / samples
    except:
        pass
    return None

def create_excel_with_baselines(df_quant, fp32_baselines):
    """Create Excel sheet with proper FP32 baselines"""
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    
    # Define headers
    headers = ['Dataset', 'Method', 'Bits', 'Avg Inference Time/Sample', 'Avg Speed Up']
    
    for seed in [42, 123, 7]:
        for n in [100, 500, 1000]:
            headers.append(f'Seed{seed}_N{n}')
    
    # Write headers
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.font = Font(color="FFFFFF", bold=True, size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Methods to include
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
    
    for method, bits in methods_order:
        method_data = df_quant[(df_quant['method'] == method) & (df_quant['bits'] == bits)]
        
        if method_data.empty and method != 'fp32':
            continue
        
        # Calculate per-sample times
        per_sample_times = []
        seed_n_data = {}
        
        for _, row in method_data.iterrows():
            seed = int(row['seed'])
            n = int(row['n'])
            per_sample = calculate_per_sample_time(row)
            
            if per_sample is not None:
                per_sample_times.append(per_sample)
                seed_n_data[(seed, n)] = per_sample
        
        # Average per-sample time
        avg_per_sample = sum(per_sample_times) / len(per_sample_times) if per_sample_times else None
        
        # Calculate speed up (relative to FP32 average)
        fp32_times = [t for (s, n), t in fp32_baselines.items()]
        fp32_avg = sum(fp32_times) / len(fp32_times) if fp32_times else None
        
        if method == 'fp32':
            avg_speedup = 1.0
        elif avg_per_sample and fp32_avg:
            avg_speedup = fp32_avg / avg_per_sample
        else:
            avg_speedup = None
        
        # Write row
        col = 1
        
        # Dataset
        ws.cell(row=row_num, column=col, value='Delicious200k')
        col += 1
        
        # Method
        method_display = method.replace('_', ' ').title()
        ws.cell(row=row_num, column=col, value=method_display)
        col += 1
        
        # Bits
        ws.cell(row=row_num, column=col, value=bits)
        col += 1
        
        # Average inference time per sample
        if avg_per_sample:
            ws.cell(row=row_num, column=col, value=round(avg_per_sample, 6))
        col += 1
        
        # Average speed up
        if avg_speedup:
            ws.cell(row=row_num, column=col, value=round(avg_speedup, 4))
        col += 1
        
        # Add seed/n results
        for seed in [42, 123, 7]:
            for n in [100, 500, 1000]:
                if (seed, n) in seed_n_data:
                    per_sample = seed_n_data[(seed, n)]
                    fp32_baseline = fp32_baselines.get((seed, n))
                    
                    if method == 'fp32':
                        value_str = f"{round(per_sample, 6)}"
                    elif fp32_baseline:
                        speedup = fp32_baseline / per_sample
                        value_str = f"{round(per_sample, 6)} ({round(speedup, 2)}x)"
                    else:
                        value_str = f"{round(per_sample, 6)}"
                    
                    ws.cell(row=row_num, column=col, value=value_str)
                else:
                    # Check for fallback (clip_mixed -> use sym)
                    if '_clip_mixed' in method and seed == 42 and n == 100:
                        fallback_method = method.replace('_clip_mixed', '')
                        fallback_data = df_quant[(df_quant['method'] == fallback_method) & 
                                               (df_quant['bits'] == bits) &
                                               (df_quant['seed'] == seed) &
                                               (df_quant['n'] == n)]
                        if not fallback_data.empty:
                            per_sample = calculate_per_sample_time(fallback_data.iloc[0])
                            fp32_baseline = fp32_baselines.get((seed, n))
                            if per_sample and fp32_baseline:
                                speedup = fp32_baseline / per_sample
                                value_str = f"{round(per_sample, 6)} ({round(speedup, 2)}x)*"
                                ws.cell(row=row_num, column=col, value=value_str)
                
                col += 1
        
        # Format row
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=row_num, column=c)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            if row_num % 2 == 0:
                cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
        
        row_num += 1
    
    # Set column widths
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['C'].width = 8
    ws.column_dimensions['D'].width = 22
    ws.column_dimensions['E'].width = 15
    
    for col in range(6, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 22
    
    # Freeze header
    ws.freeze_panes = 'A2'
    
    # Save
    output_file = "/DATA1/rudra1/dismec_quantization/dismecpp/results/inference_timing/Delicious200k_Inference_Timing_Results.xlsx"
    wb.save(output_file)
    
    print(f"\n✅ Excel file created: {output_file}")
    return output_file

def main():
    print("=" * 70)
    print("Creating Excel Report with FP32 Baselines")
    print("=" * 70)
    
    df_baseline, df_quant = read_data_files()
    fp32_baselines = extract_fp32_baselines(df_baseline)
    
    if not fp32_baselines:
        print("⚠️  No FP32 baselines found!")
        return
    
    create_excel_with_baselines(df_quant, fp32_baselines)
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
