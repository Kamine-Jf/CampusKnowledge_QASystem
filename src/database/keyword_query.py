# -*- coding: utf-8 -*-
"""校园问答系统 - 关键词查询执行脚本"""

import sys
import os

# 将项目src目录加入Python路径，确保能导入database模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database.mysql_conn import MysqlConnection
from database.db_operate import query_by_keyword, excel_to_mysql
from database.test_db_full import _execute_sql_file, _reset_business_tables


def init_database():
    """初始化数据库（创建库表+导入Excel数据），首次运行需执行"""
    # 1. 验证MySQL连接
    mysql_client = MysqlConnection()
    if not mysql_client.connect_db(use_database=False):
        raise ConnectionError("MySQL连接失败，请检查src/database/mysql_config.py配置")
    mysql_client.close_db()

    # 2. 执行建表SQL
    sql_file_path = os.path.join("src", "database", "create_tables.sql")
    _execute_sql_file(sql_file_path)
    print("✅ 数据库表创建完成")

    # 3. 导入Excel数据（需确保data/structured_data目录下有Excel文件）
    insert_count = excel_to_mysql()
    print(f"✅ Excel数据导入完成，共导入 {insert_count} 条记录")


def run_keyword_query(keyword: str):
    """执行关键词查询并打印结果"""
    if not keyword:
        print("❌ 关键词不能为空，请输入查询关键词")
        return

    # 执行关键词模糊查询
    results = query_by_keyword(keyword)

    # 打印查询结果
    print(f"\n🔍 关键词「{keyword}」查询结果（共 {len(results)} 条）：")
    print("-" * 80)
    for idx, record in enumerate(results, 1):
        print(f"\n【{idx}】")
        print(f"分类：{record['category']}")
        print(f"事项：{record['item']}")
        print(f"操作步骤：{record['operation'][:100]}..." if len(record['operation']) > 100 else f"操作步骤：{record['operation']}")
        print(f"时间要求：{record['time_requirement']}")
        print(f"办理渠道：{record['channel']}")
        print(f"备注：{record['source_note'][:50]}..." if len(record['source_note']) > 50 else f"备注：{record['source_note']}")
        print("-" * 80)


if __name__ == "__main__":
    # 首次运行请取消下方注释，初始化数据库（创建表+导入Excel数据）
    init_database()

    # 输入要查询的关键词（示例：缓考、奖学金、公共选课）
    target_keyword = input("请输入查询关键词：").strip()
    run_keyword_query(target_keyword)