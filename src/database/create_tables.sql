-- 初始化校园问答数据库，确保使用 utf8mb4 编码以支持中文及特殊符号。
CREATE DATABASE IF NOT EXISTS campus_qa_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 切换至核心业务数据库。
USE campus_qa_db;

-- 创建结构化校园业务数据表，对应 Excel 六列数据。
CREATE TABLE IF NOT EXISTS campus_struct_data (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID，自增',
    category VARCHAR(50) NOT NULL COMMENT '大类分类，如：教务流程、奖助勤贷、校园服务',
    item VARCHAR(100) NOT NULL COMMENT '具体事项，如：公共选课、缓考申请、国家奖学金申请',
    operation TEXT NOT NULL COMMENT '详细操作步骤，长文本存储，防止内容截断',
    time_requirement VARCHAR(100) DEFAULT '' COMMENT '时间要求，如：每年9月、考前一周',
    channel VARCHAR(100) DEFAULT '' COMMENT '办理渠道，如：教务系统、智慧学工APP',
    source_note TEXT NOT NULL COMMENT '信息来源/备注，存储备注、链接、标准等信息'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='校园结构化业务数据表';

-- 创建用户查询历史表，记录问答全过程。
CREATE TABLE IF NOT EXISTS user_query_history (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID，自增',
    session_id VARCHAR(64) NOT NULL DEFAULT 'default' COMMENT '会话ID，用于上下文记忆与历史追踪',
    query_content VARCHAR(500) NOT NULL COMMENT '用户查询的原始内容',
    answer_content TEXT NOT NULL COMMENT '系统返回的答复内容',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '查询时间',
    INDEX idx_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户问答历史记录表';

-- 创建用户认证表，支持用户注册与登录。
CREATE TABLE IF NOT EXISTS user_auth (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID，自增',
    name VARCHAR(50) NOT NULL COMMENT '用户姓名',
    student_id VARCHAR(30) NOT NULL COMMENT '学号',
    phone VARCHAR(20) NOT NULL COMMENT '手机号',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
    UNIQUE KEY uk_student_id (student_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户登录注册信息表';

-- 兼容已存在旧表：若历史库中尚无 session_id 字段，请执行以下迁移语句。
ALTER TABLE user_query_history
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(64) NOT NULL DEFAULT 'default' COMMENT '会话ID，用于上下文记忆与历史追踪';

-- 兼容已存在旧表：若未建索引，请执行以下语句（可能因重复索引报错，可忽略）。
CREATE INDEX idx_session_id ON user_query_history(session_id);

-- 创建 PDF 文档信息表，供后续文档管理使用。
CREATE TABLE IF NOT EXISTS pdf_doc_info (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID，自增',
    doc_name VARCHAR(200) NOT NULL COMMENT 'PDF 文档名称',
    doc_path VARCHAR(500) NOT NULL COMMENT 'PDF 在文件系统中的存储路径',
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='PDF 文档预留信息表';
