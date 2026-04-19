# MOODLE-11
NLP Preprocessing & Evaluation: NLTK vs. spaCy

Dự án này tập trung vào việc so sánh và đánh giá hai thư viện xử lý ngôn ngữ tự nhiên hàng đầu là NLTK và spaCy thông qua các kỹ thuật tiền xử lý văn bản cơ bản.
📌 Tính năng chính (Key Features)

Chương trình thực hiện quy trình xử lý văn bản qua 7 bước chuẩn hóa:

    Input Text: Tiếp nhận văn bản thô.

    Tokenization: Tách văn bản thành các từ/ký hiệu riêng lẻ.

    Lowercasing: Chuyển đổi toàn bộ văn bản về chữ thường.

    Stopword Removal: Loại bỏ các từ dừng (các từ phổ biến không mang nhiều ý nghĩa như "the", "is", "a").

    Punctuation Removal: Loại bỏ các dấu câu.

    Stemming: Cắt tỉa từ về dạng gốc (chỉ áp dụng với NLTK).

    Lemmatization: Đưa từ về dạng nguyên thể dựa trên từ điển.

📊 Đánh giá hiệu năng (Evaluation Metrics)

Để so sánh độ chính xác của NLTK và spaCy, mã nguồn sử dụng tập dữ liệu chuẩn (Gold Data) để tính toán các chỉ số:

    Accuracy: Độ chính xác tổng thể.

    Precision: Độ chính xác trong các kết quả dự đoán.

    Recall: Khả năng bao phủ các từ khóa cần thiết.

    F1-score: Chỉ số trung bình hài hòa giữa Precision và Recall.

Công thức tính F1-score được áp dụng:
F1=2⋅Precision+RecallPrecision⋅Recall​
🛠 Cài đặt (Installation)

    Clone repository:
    Bash

    git clone [Link-GitHub-Của-Bạn]

    Cài đặt các thư viện cần thiết:
    Bash

    pip install -r requirements.txt

    Tải dữ liệu NLTK (nếu chưa có):
    Mở Python terminal và chạy:
    Python

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

📂 Cấu trúc thư mục

    nlp_task.py: Mã nguồn chính xử lý dữ liệu và đánh giá.

    requirements.txt: Danh sách các thư viện phụ thuộc (NLTK 3.9.4, spaCy 3.8.14, v.v.).
