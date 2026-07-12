"""执行生成 400M 中表的脚本。

用法:
    python scripts/build_interim_table.py [--config configs/config.yaml]
"""

import argparse
from src.utils import load_config, setup_logging
from src.big_data_processor import BigDataProcessor


def main():
    parser = argparse.ArgumentParser(description="从原始数据生成中表")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    processor = BigDataProcessor(config_path=args.config)
    output_path = processor.run()
    print(f"中表已生成: {output_path}")


if __name__ == "__main__":
    main()