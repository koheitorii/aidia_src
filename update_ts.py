import subprocess
import os
import glob

def find_python_files():
    """translate.proのSOURCESに相当するPythonファイルを検索"""
    # 一般的なパターンでPythonファイルを検索
    python_files = []
    
    # カレントディレクトリとサブディレクトリからPythonファイルを検索
    for pattern in ['aidia/*.py', 'aidia/**/*.py', 'aidia/widgets/*.py', 'aidia/widgets/**/*.py']:
        python_files.extend(glob.glob(pattern, recursive=True))
    
    # __pycache__や.envなどの不要なディレクトリを除外
    excluded_dirs = ['__pycache__', 'env', 'build', 'dist', 'Output']
    filtered_files = []
    
    for file in python_files:
        if not any(excluded in file for excluded in excluded_dirs):
            filtered_files.append(file)
    
    return filtered_files


def update_translations():
    """pylupdate6を使用して翻訳ファイルを更新"""
    # Pythonファイルのリストを取得
    python_files = find_python_files()
    
    if not python_files:
        print("No Python files found for translation.")
        return
    
    # 翻訳ファイルのディレクトリ
    translate_dir = os.path.join('aidia', 'translate')
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(translate_dir):
        os.makedirs(translate_dir)
    
    # 翻訳ファイルのパス
    ts_file = os.path.join(translate_dir, 'ja_JP.ts')
    
    # pylupdate6コマンドを構築
    cmd = ['pylupdate6'] + python_files + ['-ts', ts_file]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # pylupdate6を実行
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"Successfully updated translation file: {ts_file}")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print(f"Error updating translation file: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: pylupdate6 not found. Make sure PyQt6 development tools are installed.")
        print("Install with: pip install PyQt6-tools")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    return True


def main():
    update_translations()


if __name__ == '__main__':
    main()