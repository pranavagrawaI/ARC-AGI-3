from pathlib import Path
Path('wrong-module-executed').write_text('shadow')
def main():return 0
