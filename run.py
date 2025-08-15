#!/usr/bin/env python3
"""
Qualcomm AI Hub 基本操作範例
"""

import qai_hub as hub

def list_devices():
    """列出所有可用裝置"""
    try:
        devices = hub.get_devices()  # 看有哪些可用裝置名稱
        print(f"找到 {len(devices)} 個可用裝置\n")
        
        # 按類型分組顯示
        device_types = {}
        for device in devices:
            device_format = next((attr.split(':')[1] for attr in device.attributes if attr.startswith('format:')), 'unknown')
            if device_format not in device_types:
                device_types[device_format] = []
            device_types[device_format].append(device)
        
        # 顯示每種類型的裝置
        for device_type, devices_list in device_types.items():
            print(f"📱 {device_type.upper()} 裝置 ({len(devices_list)} 個):")
            for device in devices_list[:5]:  # 只顯示前5個
                frameworks = [attr.split(':')[1] for attr in device.attributes if attr.startswith('framework:')]
                print(f"  • {device.name} (OS: {device.os}, 框架: {', '.join(frameworks)})")
            if len(devices_list) > 5:
                print(f"  ... 還有 {len(devices_list) - 5} 個裝置")
            print()
        
        print("API token 已正確配置！")
        return devices
        
    except Exception as e:
        print("錯誤:", e)
        print("可能需要配置 API token:")
        print("qai-hub configure --api_token YOUR_API_TOKEN")
        return None

def recommend_devices():
    """推薦適合的裝置"""
    print("🔥 推薦裝置 (適合模型部署):")
    
    recommended = [
        "Samsung Galaxy S24 (Family)",
        "Samsung Galaxy S23 (Family)",
        "Google Pixel 8 (Family)",
        "Snapdragon X Elite CRD",
        "Samsung Galaxy S25 (Family)"
    ]
    
    try:
        devices = hub.get_devices()
        for rec_name in recommended:
            matching = [d for d in devices if d.name == rec_name]
            if matching:
                device = matching[0]
                print(f"  ⭐ {device.name}")
                print(f"     • OS: {device.os}")
                chipset = next((attr for attr in device.attributes if attr.startswith('chipset:') and not attr.endswith('-proxy')), '')
                if chipset:
                    print(f"     • 晶片: {chipset.replace('chipset:', '').replace('-', ' ').title()}")
                frameworks = [attr.split(':')[1] for attr in device.attributes if attr.startswith('framework:')]
                print(f"     • 支援框架: {', '.join(frameworks)}")
                print()
    except Exception as e:
        print(f"無法獲取推薦裝置: {e}")

def main():
    print("=== Qualcomm AI Hub 裝置檢查 ===\n")
    
    devices = list_devices()
    if devices:
        print("\n" + "="*50 + "\n")
        recommend_devices()
        
        print("\n📋 可用的操作:")
        print("• 運行 'python test_connection.py' 進行完整測試")
        print("• 運行 'python cloud_simple.py' 在 Snapdragon X Elite CRD 雲端測試會議 Prompt")
        print("• 運行 'python cloud_simple.py --interactive' 互動式雲端測試")
        print("• 運行 'python taide_example.py' 部署 TAIDE 模型")
        print("• 查看 https://app.aihub.qualcomm.com/jobs 監控工作狀態")

if __name__ == "__main__":
    main()