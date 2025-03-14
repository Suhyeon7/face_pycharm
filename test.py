import mysql.connector
import json

# MySQL 데이터베이스 연결
conn = mysql.connector.connect(
    host="localhost",  # MySQL 서버 주소
    user="face",  # MySQL 사용자
    password="1234",  # MySQL 비밀번호
    database="my_face"  # 연결할 데이터베이스 이름
)

# 커서 생성
cursor = conn.cursor()

# SQL 쿼리 실행 (예: 테이블 생성)
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    embedding_vector JSON
)
""")

# 임베딩 벡터 예시 데이터
embedding_vector = json.dumps([0.25, 0.5, 0.75, 1.0])  # JSON 형식으로 변환

# 데이터 삽입
cursor.execute("INSERT INTO users (name, age, embedding_vector) VALUES (%s, %s, %s)",
               ("John Doe", 30, embedding_vector))

# 커밋 (변경사항을 데이터베이스에 반영)
conn.commit()

# 데이터 조회
cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

# 연결 종료
cursor.close()
conn.close()
