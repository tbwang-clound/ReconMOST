import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm

def get_en4_reference_grid(en4_source_dir):
    """
    从EN4数据目录中读取一个样本文件，并提取其空间坐标作为参考网格。
    创建的参考网格将不包含时间坐标。
    """
    print("="*20 + " 1. 从EN4数据创建参考网格 " + "="*20)
    en4_files = glob.glob(os.path.join(en4_source_dir, "*.nc"))
    if not en4_files:
        print(f"错误：在EN4目录 '{en4_source_dir}' 中找不到任何 .nc 文件来创建参考网格。")
        return None
    
    sample_file = en4_files[0]
    print(f"使用文件 '{os.path.basename(sample_file)}' 作为参考标准。")
    
    try:
        with xr.open_dataset(sample_file) as ds:
            # 统一坐标名称为 lev, lat, lon
            rename_dict = {}
            if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
            if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
            if 'depth' in ds.coords: rename_dict['depth'] = 'lev'
            ds_renamed = ds.rename(rename_dict)
            
            # **核心修正**：创建一个只包含空间坐标的参考网格
            spatial_coords = {
                'lev': ds_renamed.lev,
                'lat': ds_renamed.lat,
                'lon': ds_renamed.lon
            }
            reference_grid = xr.Dataset(coords=spatial_coords)
            
            print(f"参考网格创建成功，维度为 (lev: {len(reference_grid.lev)}, lat: {len(reference_grid.lat)}, lon: {len(reference_grid.lon)})")
            return reference_grid
    except Exception as e:
        print(f"创建参考网格时出错: {e}")
        return None



def process_and_regrid_files(source_dir, dest_dir, reference_grid, is_training_data):
    """
    通用处理函数：打开文件，统一变量/单位，插值到参考网格，然后保存。
    如果是训练数据，则额外进行时间分割。
    """
    os.makedirs(dest_dir, exist_ok=True)
    source_files = glob.glob(os.path.join(source_dir, "*.nc"))
    
    if not source_files:
        print(f"警告：在源目录 '{source_dir}' 中未找到任何 .nc 文件。")
        return

    print(f"\n--- 开始处理目录: {source_dir} ---")
    
    for file_path in source_files:
        print(f"  > 正在处理: {os.path.basename(file_path)}")
        try:
            with xr.open_dataset(file_path, chunks={'time': 12}) as ds:
                # **步骤A: 统一所有文件的维度和坐标名称为标准形式**
                rename_dict = {}
                # 检查并重命名维度
                for dim_name in ['latitude', 'longitude', 'depth']:
                    if dim_name in ds.dims:
                        rename_dict[dim_name] = dim_name[0:3] # latitude -> lat
                # 检查并重命名坐标
                for coord_name in ['latitude', 'longitude', 'depth']:
                    if coord_name in ds.coords:
                        rename_dict[coord_name] = coord_name[0:3]
                if rename_dict:
                    ds = ds.rename(rename_dict)
                
                # **步骤B: 统一数据变量名和单位**
                var_name = 'thetao'
                potential_names = ['thetao', 'temperature', 'potential_temperature', 'temp']
                current_var_name = next((name for name in potential_names if name in ds.data_vars), None)
                if not current_var_name:
                    print(f"    错误：找不到温度变量。跳过。")
                    continue
                if current_var_name != var_name:
                    ds = ds.rename({current_var_name: var_name})
                if 'units' in ds[var_name].attrs and ds[var_name].attrs['units'].lower() in ['k', 'kelvin']:
                    ds[var_name] = ds[var_name] - 273.15
                    ds[var_name].attrs['units'] = 'degC'

                # **步骤C: (核心) 将数据插值到EN4参考网格**
                print(f"    正在插值到EN4参考网格...")
                ds_regridded = ds.interp_like(reference_grid, method='linear')
                
                # **步骤D: 根据数据类型进行保存**
                if is_training_data: # CMIP6: 分割并保存
                    num_timesteps = len(ds_regridded.time)
                    print(f"    文件包含 {num_timesteps} 个时间步，开始分割...")
                    for i in tqdm(range(num_timesteps), desc=f"    分割中"):
                        single_month_ds = ds_regridded.isel(time=i)
                        time_val = single_month_ds.time.dt
                        # 从原始文件名中提取模型标识
                        model_id = "_".join(os.path.basename(file_path).split('_')[2:6])
                        new_filename = f"{model_id}_{time_val.year.item():04d}{time_val.month.item():02d}.nc"
                        output_path = os.path.join(dest_dir, new_filename)
                        single_month_ds.to_netcdf(output_path)
                else: # EN4: 直接保存
                    output_path = os.path.join(dest_dir, os.path.basename(file_path))
                    ds_regridded.to_netcdf(output_path)

        except Exception as e:
            print(f"    处理文件 {file_path} 时发生严重错误: {e}")

if __name__ == '__main__':
    # --- 1. 配置路径 ---
    SOURCE_TRAIN_DIR = "/data/coding/data/ReconMOST_train"
    SOURCE_TEST_DIR = "/data/coding/data/ReconMOST-testdata"
    PROCESSED_TRAIN_DIR = "/data/coding/data/ReconMOST_train_processed_en4grid"
    PROCESSED_TEST_DIR = "/data/coding/data/ReconMOST_testdata_processed_en4grid"

    # --- 2. 从EN4数据创建参考网格 ---
    ref_grid = get_en4_reference_grid(SOURCE_TEST_DIR)

    # --- 修正之处：将 "if ref_grid:" 改为 "if ref_grid is not None:" ---
    if ref_grid is not None:
        # --- 3. 处理训练数据 (CMIP6)，使其匹配EN4网格 ---
        process_and_regrid_files(SOURCE_TRAIN_DIR, PROCESSED_TRAIN_DIR, ref_grid, is_training_data=True)
        
        # --- 4. 处理测试数据 (EN4)，确保其坐标和变量名等完全标准化 ---
        process_and_regrid_files(SOURCE_TEST_DIR, PROCESSED_TEST_DIR, ref_grid, is_training_data=False)
        
        print("\n*** 所有数据预处理完成！***")
    else:
        # 这个分支现在只会在 get_en4_reference_grid 函数真正失败时执行
        print("\n预处理失败，因为无法从EN4数据创建参考网格。请检查路径和文件。")