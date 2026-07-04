"""
Git 版本历史查询工具
用于查看 web_server.py 在 Git 仓库中的历史版本内容

该脚本通过调用 Git 命令行，获取指定文件（server/web_server.py）的历史提交记录，
并对比当前版本与历史版本的差异，可用于恢复或审查代码变更。
"""

import subprocess

def run():
    """从 Git 仓库中查询 web_server.py 的历史版本，并打印相关信息。"""

    try:
        # 获取 web_server.py 在 HEAD 版本（当前最新提交）的内容
        output = subprocess.check_output(["git", "show", "HEAD:server/web_server.py"], cwd="/home/value/Keshe/fire").decode('utf-8')
        # 查看最近的 10 条提交记录（简洁格式）
        log = subprocess.check_output(["git", "log", "-n", "10", "--oneline"], cwd="/home/value/Keshe/fire").decode('utf-8')
        print("GIT LOG:")
        print(log)
        
        # 查看 web_server.py 文件的所有历史提交记录（按时间顺序）
        commits = subprocess.check_output(["git", "log", "--follow", "--oneline", "server/web_server.py"], cwd="/home/value/Keshe/fire").decode('utf-8')
        print("COMMITS FOR web_server.py:")
        print(commits)
        
        # 查看 HEAD~4（当前版本往前 4 个提交）中的 web_server.py 内容
        # 用于对比当前版本与历史版本的差异，便于恢复或分析修改
        original_content = subprocess.check_output(["git", "show", "HEAD~4:server/web_server.py"], cwd="/home/value/Keshe/fire").decode('utf-8')
        # 在历史版本中查找 DASHBOARD_TEMPLATE 模板，以便恢复或对比
        start_idx = original_content.find("DASHBOARD_TEMPLATE")
        if start_idx != -1:
            print("\nORIGINAL DASHBOARD_TEMPLATE FOUND:")
            print(original_content[start_idx:start_idx+1500])
        else:
            print("\nDASHBOARD_TEMPLATE not found in HEAD~4")
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    run()