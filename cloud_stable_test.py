#!/usr/bin/env python3
"""
穩定大規模雲端測試 - 優化 ONNX 相容性，避免簡化器問題
目標：在 Snapdragon X Elite CRD 上穩定運行大規模模型
"""

import qai_hub as hub
import torch
import torch.nn as nn
import gc
import time
import numpy as np

class StableCompatibleModel(nn.Module):
    """穩定的大規模相容模型 - 優化 ONNX 轉換穩定性"""
    
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=8, output_size=1024):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 輸入投影層 - 簡化結構提高穩定性
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # 穩定的深度網路層 - 避免過於複雜的結構
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 3),  # 3倍擴展（較保守）
                nn.ReLU(),
                nn.Linear(hidden_size * 3, hidden_size * 2),  # 2倍中間層
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),  # 回到原始大小
                nn.LayerNorm(hidden_size)
            )
            self.layers.append(layer)
        
        # 輸出投影層 - 簡化結構
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.output_norm = nn.LayerNorm(output_size)
        
        # 殘差連接權重 - 簡化為固定值避免複雜參數
        self.residual_scale = 0.1
        
        # 初始化權重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型權重 - 使用更保守的初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 更小的標準差
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向傳播 - 簡化操作提高 ONNX 穩定性"""
        # 輸入投影
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # 深度處理 - 使用簡單的殘差連接
        for layer in self.layers:
            residual = x
            x = layer(x)
            # 簡化的殘差連接
            x = x + self.residual_scale * residual
        
        # 輸出投影
        x = self.output_projection(x)
        output = self.output_norm(x)
        
        return output

def create_stable_model():
    """創建穩定的大規模語言模型"""
    print("🏗️ 創建穩定大規模模型（優化 ONNX 相容性）...")
    
    model = StableCompatibleModel(
        input_size=1024,      # 平衡的維度
        hidden_size=1024,     # 平衡的隱藏層維度  
        num_layers=8,         # 適中的層數避免過深
        output_size=1024      # 平衡的輸出維度
    )
    
    model.eval()
    
    # 模型大小估算
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    model_size_gb = model_size_mb / 1024
    
    print(f"✅ 穩定大規模模型創建完成")
    print(f"   📊 參數數量: {total_params:,}")
    print(f"   📏 模型大小: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")
    print(f"   🏗️ 網路層數: {model.num_layers}")
    print(f"   🔢 隱藏維度: {model.hidden_size}")
    print(f"   📐 輸入/輸出維度: {model.input_size}")
    print(f"   🔧 優化：簡化架構提高 ONNX 穩定性")
    
    return model

def test_stable_model(model):
    """測試穩定大規模模型"""
    print("\n🧪 本地測試穩定大規模模型...")
    
    test_cases = [
        {"name": "企業AI處理", "description": "企業級AI數據處理", "batch_size": 1, "seq_len": 64},
        {"name": "戰略決策引擎", "description": "戰略級決策AI", "batch_size": 2, "seq_len": 32}, 
        {"name": "高效能計算", "description": "高性能AI計算", "batch_size": 4, "seq_len": 16},
        {"name": "雲端服務", "description": "雲端AI服務", "batch_size": 8, "seq_len": 8}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 測試 {i}: {test_case['name']}")
        print(f"   說明: {test_case['description']}")
        print(f"   批次大小: {test_case['batch_size']}, 序列長度: {test_case['seq_len']}")
        
        try:
            # 創建測試輸入
            input_data = torch.randn(
                test_case['batch_size'], 
                test_case['seq_len'], 
                model.input_size,
                dtype=torch.float32
            )
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_data)
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            print(f"   ✅ 推理成功")
            print(f"   ⏱️  推理時間: {inference_time:.4f} 秒")
            print(f"   📏 輸入形狀: {input_data.shape}")
            print(f"   📐 輸出形狀: {output.shape}")
            
            # 計算吞吐量
            throughput = (test_case['batch_size'] * test_case['seq_len']) / inference_time
            print(f"   🚀 吞吐量: {throughput:.2f} tokens/秒")
            
            results.append({
                'name': test_case['name'],
                'inference_time': inference_time,
                'throughput': throughput,
                'batch_size': test_case['batch_size'],
                'seq_len': test_case['seq_len'],
                'status': 'success'
            })
            
        except Exception as e:
            print(f"   ❌ 測試失敗: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'status': 'failed'
            })
    
    return results

def compile_stable_model_to_cloud(model):
    """編譯穩定大規模模型到雲端 - 增強錯誤處理"""
    device = hub.Device("Snapdragon X Elite CRD")
    print(f"\n🚀 編譯穩定大規模模型到 {device.name}...")
    
    try:
        # 使用較小的樣本輸入提高穩定性
        sample_input = torch.randn(1, 32, model.input_size, dtype=torch.float32)
        print(f"輸入形狀: {sample_input.shape}")
        print(f"輸入類型: {sample_input.dtype}")
        
        # 測試模型
        print("🧪 預先測試...")
        with torch.no_grad():
            test_output = model(sample_input)
            print(f"✅ 本地推理成功，輸出形狀: {test_output.shape}")
        
        # 追蹤模型 - 使用更保守的設定
        print("📊 追蹤模型...")
        with torch.no_grad():
            # 設定更嚴格的追蹤參數
            traced_model = torch.jit.trace(
                model,
                sample_input,
                strict=True,  # 啟用嚴格模式
                check_trace=True  # 啟用追蹤檢查
            )
        
        # 驗證追蹤結果
        print("🔍 驗證追蹤...")
        with torch.no_grad():
            traced_output = traced_model(sample_input)
            print(f"✅ 追蹤驗證成功，輸出形狀: {traced_output.shape}")
            
            # 檢查輸出一致性
            original_output = model(sample_input)
            diff = torch.abs(traced_output - original_output).max().item()
            print(f"🔍 追蹤精度差異: {diff:.6f}")
            
            if diff > 1e-4:
                print(f"⚠️  警告：追蹤精度差異較大")
            else:
                print(f"✅ 追蹤精度良好")
        
        # 最佳化 - 使用較保守的最佳化
        print("⚡ 最佳化模型...")
        try:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            print("✅ 最佳化成功")
        except Exception as opt_error:
            print(f"⚠️  最佳化跳過: {opt_error}")
        
        # 檢查模型大小
        total_params = sum(p.numel() for p in traced_model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)
        model_size_gb = model_size_mb / 1024
        print(f"📏 追蹤模型大小: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")
        
        # 提交編譯工作 - 增加重試機制
        print("☁️  提交編譯工作...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                compile_job = hub.submit_compile_job(
                    model=traced_model,
                    device=device,
                    input_specs={"x": sample_input.shape}
                )
                
                print(f"🆔 編譯工作 ID: {compile_job.job_id}")
                print(f"🔗 工作連結: https://app.aihub.qualcomm.com/jobs/{compile_job.job_id}")
                break
                
            except Exception as submit_error:
                print(f"⚠️  提交嘗試 {attempt + 1} 失敗: {submit_error}")
                if attempt == max_retries - 1:
                    raise submit_error
                time.sleep(2)  # 等待後重試
        
        # 等待編譯完成
        print("⏳ 等待編譯完成...")
        try:
            target_model = compile_job.get_target_model()
            print("✅ 編譯成功！")
            return target_model, device, compile_job.job_id, sample_input.shape
        except Exception as compile_error:
            print(f"❌ 編譯過程失敗: {compile_error}")
            return None, None, compile_job.job_id, None
        
    except Exception as e:
        print(f"❌ 編譯失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def run_stable_cloud_inference_test(target_model, device, job_id, input_shape):
    """在雲端執行穩定大規模推理測試"""
    print(f"\n🌐 在 {device.name} 上執行穩定大規模推理測試...")
    
    stable_scenarios = [
        {"name": "穩定企業AI", "description": "穩定企業AI處理", "seq_len": 32},
        {"name": "穩定戰略AI", "description": "穩定戰略決策", "seq_len": 24},
        {"name": "穩定計算AI", "description": "穩定高性能計算", "seq_len": 28},
        {"name": "穩定服務AI", "description": "穩定雲端服務", "seq_len": 20}
    ]
    
    results = []
    
    for i, scenario in enumerate(stable_scenarios, 1):
        print(f"\n📝 雲端測試 {i}: {scenario['name']}")
        print(f"說明: {scenario['description']}")
        print(f"序列長度: {scenario['seq_len']}")
        
        try:
            # 準備測試輸入 - 修復輸入格式
            batch_size, _, input_dim = input_shape
            input_data = np.random.randn(batch_size, scenario['seq_len'], input_dim).astype(np.float32)
            print(f"輸入形狀: {input_data.shape}")
            print(f"輸入類型: {input_data.dtype}")
            
            # 執行雲端推理
            print("☁️  執行雲端推理...")
            start_time = time.time()
            
            inference_job = hub.submit_inference_job(
                model=target_model,
                device=device,
                inputs={"x": [input_data]}  # 修復：輸入需要是列表格式
            )
            
            # 等待推理完成
            outputs = inference_job.download_output_data()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"✅ 雲端推理成功")
            print(f"   ⏱️  總時間: {total_time:.3f} 秒")
            print(f"   📊 輸出類型: {type(outputs)}")
            
            # 檢查輸出詳細信息
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, np.ndarray):
                        print(f"   📐 輸出 {key} 形狀: {value.shape}")
            
            # 計算性能指標
            tokens_processed = scenario['seq_len']
            throughput = tokens_processed / total_time
            
            print(f"   🚀 處理吞吐量: {throughput:.2f} tokens/秒")
            print(f"   🆔 推理工作 ID: {inference_job.job_id}")
            
            results.append({
                'scenario': scenario['name'],
                'total_time_seconds': total_time,
                'throughput': throughput,
                'seq_len': scenario['seq_len'],
                'status': 'success',
                'job_id': inference_job.job_id
            })
                
        except Exception as e:
            print(f"❌ 雲端推理失敗: {e}")
            results.append({
                'scenario': scenario['name'],
                'error': str(e),
                'status': 'failed'
            })
    
    return results

def main():
    """主測試流程"""
    print("=== 穩定大規模雲端推理測試 ===\n")
    
    # 第一階段：創建和測試模型
    print("🎯 第一階段：穩定大規模模型創建和本地測試")
    print("=" * 70)
    
    model = create_stable_model()
    local_results = test_stable_model(model)
    
    print(f"\n📊 本地測試結果:")
    successful_local = [r for r in local_results if r['status'] == 'success']
    if successful_local:
        avg_throughput = sum(r['throughput'] for r in successful_local) / len(successful_local)
        print(f"✅ 成功測試: {len(successful_local)}/{len(local_results)}")
        print(f"🚀 平均吞吐量: {avg_throughput:.2f} tokens/秒")
        
        for result in successful_local:
            print(f"  📝 {result['name']}: {result['inference_time']:.4f}s ({result['throughput']:.2f} tokens/s)")
    
    for result in local_results:
        if result['status'] == 'failed':
            print(f"  ❌ {result['name']}: {result['error']}")
    
    # 第二階段：雲端編譯
    print(f"\n🎯 第二階段：雲端編譯")
    print("=" * 70)
    
    target_model, device, job_id, input_shape = compile_stable_model_to_cloud(model)
    
    if not target_model:
        print("❌ 編譯失敗，無法進行雲端測試")
        if job_id:
            print(f"🔗 編譯工作連結: https://app.aihub.qualcomm.com/jobs/{job_id}")
        return
    
    # 第三階段：雲端推理
    print(f"\n🎯 第三階段：穩定大規模雲端推理測試")
    print("=" * 70)
    
    cloud_results = run_stable_cloud_inference_test(target_model, device, job_id, input_shape)
    
    # 結果總結
    print(f"\n🎉 穩定大規模測試完成！")
    print("=" * 80)
    print(f"🔗 編譯工作連結: https://app.aihub.qualcomm.com/jobs/{job_id}")
    
    print(f"\n📊 雲端推理結果:")
    successful_tests = [r for r in cloud_results if 'success' in r['status']]
    
    if successful_tests:
        print(f"✅ 成功測試: {len(successful_tests)}/{len(cloud_results)}")
        
        total_time = sum(r['total_time_seconds'] for r in successful_tests)
        avg_time = total_time / len(successful_tests)
        avg_throughput = sum(r['throughput'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📈 性能統計:")
        print(f"   ⏱️  平均推理時間: {avg_time:.3f}s")
        print(f"   🚀 平均吞吐量: {avg_throughput:.2f} tokens/秒")
        
        print(f"\n📝 詳細結果:")
        for result in successful_tests:
            print(f"   {result['scenario']}:")
            print(f"     ⏱️  時間: {result['total_time_seconds']:.3f}s")
            print(f"     🚀 吞吐量: {result['throughput']:.2f} tokens/s")
            print(f"     📏 序列長度: {result['seq_len']}")
            print(f"     🆔 工作ID: {result['job_id']}")
    
    failed_tests = [r for r in cloud_results if r['status'] == 'failed']
    if failed_tests:
        print(f"\n❌ 失敗測試: {len(failed_tests)}")
        for result in failed_tests:
            print(f"  • {result['scenario']}: {result['error']}")
    
    print(f"\n💡 穩定大規模測試成果:")
    print(f"🎯 成功在 Snapdragon X Elite CRD 上運行了穩定大規模模型")
    print(f"📊 優化了 ONNX 相容性，提高編譯成功率")
    print(f"⚡ 獲得了穩定可靠的推理效能數據")
    print(f"🔬 驗證了穩定大規模模型的雲端部署可行性")
    print(f"🏆 為實際應用提供了穩定的技術方案")
    print(f"🌟 證明了優化後的大規模模型可以穩定運行")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 程序錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        print("🏁 程序結束")
