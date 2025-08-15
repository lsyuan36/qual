# TAIDE 8B Qualcomm AI Hub 部署專案

這個專案展示如何將 TAIDE 8B 模型部署到 Qualcomm AI Hub 進行最佳化和推理。

## 📋 專案結構

```
qual/
├── models/
│   └── taide8b/          # TAIDE 8B 模型檔案
├── run.py                # 基本裝置檢查和連線測試
├── test_connection.py    # 完整的連線和功能測試
├── cloud_simple.py       # Snapdragon X Elite CRD 雲端推理測試（簡化版）
├── taide_example.py      # TAIDE 模型部署範例
├── dl.py                 # 模型下載腳本
└── README.md             # 本檔案
```

## ✅ 已完成設置

- ✅ Python 虛擬環境 (3.12.11)
- ✅ 已安裝必要套件
  - `qai-hub` (0.31.0)
  - `qai-hub-models` (0.33.1)
  - `torch`, `transformers` 等
- ✅ API Token 已配置
- ✅ TAIDE 8B 模型已下載

## 🚀 使用方法

### 1. 基本連線測試
```bash
python run.py
```
這會顯示所有可用裝置和推薦的部署目標。

### 2. 完整功能測試
```bash
python test_connection.py
```
執行完整的連線測試和相容性檢查。

### 3. 雲端會議 Prompt 測試 ⭐ **推薦**
```bash
python cloud_simple.py
```
在 Snapdragon X Elite CRD 雲端硬體上測試您的會議 Prompt，獲取精確的 inference time 和 peak memory 數值。

### 4. 互動式雲端測試
```bash
python cloud_simple.py --interactive
```
互動式輸入您的 prompt 進行雲端測試。

### 5. 部署 TAIDE 模型
```bash
python taide_example.py
```
將 TAIDE 8B 模型部署到 Qualcomm AI Hub 進行最佳化。

## 🎯 專門針對您的需求

### 會議 Prompt 測試
專門為您的兩個主要 prompt 設計：

1. **會議總結 Prompt**：條列式彙整會議討論要點
2. **會議議程 Prompt**：為會議腦力激盪討論議程

### 關鍵效能指標
- ⏱️ **Inference Time**：推理執行時間（秒）
- 💾 **Peak Memory**：峰值記憶體使用量（MB）

### 測試環境
- 🖥️ **Snapdragon X Elite CRD**：Windows 11, NPU 加速
- 🌐 **雲端執行**：真實硬體效能數據

## 📱 推薦的目標裝置

根據您的需求選擇：

### 🔥 最新高效能裝置
- **Samsung Galaxy S25 (Family)** - 最新 Snapdragon 8 Elite
- **Samsung Galaxy S24 (Family)** - Snapdragon 8 Gen 3
- **Samsung Galaxy S23 (Family)** - Snapdragon 8 Gen 2

### 💻 桌面/筆電裝置
- **Snapdragon X Elite CRD** - Windows 11, 支援 NPU 加速

### 🏠 IoT/嵌入式裝置
- **QCS8550 (Proxy)** - 工業級 IoT 平台

## 🔧 主要功能

### 模型編譯
- 將 PyTorch 模型轉換為 TensorFlow Lite
- 針對目標裝置最佳化
- 支援 CPU, GPU, NPU 加速

### 效能分析
- 實際裝置上的效能評測
- 延遲、吞吐量分析
- 記憶體使用量分析

### 推理測試
- 在真實裝置上執行推理
- 比較 CPU vs 最佳化版本的準確度
- 下載最佳化後的模型檔案

## 📊 支援的框架

- **TensorFlow Lite** - 大部分行動裝置
- **ONNX Runtime** - 跨平台相容性
- **Qualcomm Neural Network (QNN)** - 最佳效能 (Snapdragon 裝置)

## 🌐 有用連結

- [Qualcomm AI Hub Dashboard](https://app.aihub.qualcomm.com/jobs) - 監控工作狀態
- [官方文檔](https://app.aihub.qualcomm.com/docs) - 詳細使用指南
- [GitHub 範例](https://github.com/quic/ai-hub-models) - 更多模型範例

## 🛠️ 故障排除

### API Token 問題
如果出現認證錯誤，重新配置 API Token：
```bash
qai-hub configure --api_token YOUR_API_TOKEN
```

### 模型載入問題
確保 `models/taide8b/` 目錄包含所有必要檔案：
- `config.json`
- `tokenizer.json`
- `model-*.safetensors`

### 記憶體不足
對於大型模型，考慮：
- 使用較小的輸入序列長度
- 選擇記憶體更大的目標裝置
- 啟用模型量化

## 💡 提示

1. **首次使用**：建議先運行 `test_connection.py` 確認一切正常
2. **效能最佳化**：選擇支援 QNN 框架的 Snapdragon 裝置
3. **開發除錯**：可在 [AI Hub Dashboard](https://app.aihub.qualcomm.com/jobs) 即時監控工作進度
4. **生產部署**：下載最佳化後的模型檔案用於實際應用

## 📞 支援

如有問題，請參考：
- [Qualcomm AI Hub 文檔](https://app.aihub.qualcomm.com/docs)
- [社群論壇](https://developer.qualcomm.com/forums)
