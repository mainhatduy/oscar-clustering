# Oscar-VI Embedding Processor

Công cụ xử lý embedding cho dataset Vietnamese OSCAR sử dụng model Qwen3-Embedding-0.6B.

## Tổng quan

Script này sẽ:
- Tải từng chunk từ dataset `myduy/oscar-vi/processed_chunks`
- Xử lý embedding bằng model `Qwen/Qwen3-Embedding-0.6B`
- Lưu kết quả lên repository HuggingFace mới
- Tự động xóa dữ liệu tạm thời để tiết kiệm bộ nhớ

## Cài đặt

1. Clone repo và cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Tạo file `.env` từ template:
```bash
cp env_example.txt .env
```

3. Thêm HuggingFace token của bạn vào file `.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Cách sử dụng

### Test với một chunk đơn lẻ

Trước khi chạy toàn bộ dataset, hãy test với một chunk:

```bash
python test_single_chunk.py
```

### Xử lý toàn bộ dataset

```bash
python embedding_processor.py
```

## Cấu hình

Bạn có thể thay đổi các tham số trong file `embedding_processor.py`:

```python
processor = OscarEmbeddingProcessor(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    output_repo="myduy/oscar-vi-embeddings",  # Thay đổi tên repo output
    max_length=8192,
    embedding_dim=1024,
    batch_size=8,  # Điều chỉnh theo GPU memory
)
```

### Tham số quan trọng:

- `batch_size`: Giảm nếu gặp lỗi out of memory
- `max_length`: Độ dài tối đa của sequence
- `output_repo`: Tên repository để lưu embeddings

## Cấu trúc Output

Mỗi chunk sẽ tạo ra một file JSON với format:

```json
[
  {
    "id": "chunk_name_doc_id",
    "embedding": [0.1, 0.2, 0.3, ...]
  }
]
```

- `id`: Tên chunk + ID của document gốc
- `embedding`: Vector embedding 1024 chiều đã được normalize

## Yêu cầu hệ thống

- Python 3.8+
- GPU CUDA (khuyến nghị) hoặc CPU
- RAM: Tối thiểu 8GB, khuyến nghị 16GB+
- Dung lượng: ~50GB trống để xử lý tạm thời

## Estimate thời gian

- Với GPU Tesla V100: ~2-3 ngày
- Với GPU RTX 3090: ~4-5 ngày
- Với CPU: ~2-3 tuần

## Xử lý lỗi

Script được thiết kế để:
- Tự động retry khi gặp lỗi network
- Skip chunk bị lỗi và tiếp tục với chunk tiếp theo
- Log chi tiết để debug
- Tự động dọn dẹp memory

## Monitoring

Script sẽ log thông tin chi tiết:
- Tiến độ xử lý từng chunk
- Số lượng documents được embed
- Thời gian xử lý
- Lỗi nếu có

## Notes

1. **HF_TOKEN**: Cần có quyền write để tạo repository mới
2. **Memory**: Script tự động dọn dẹp sau mỗi chunk
3. **Resumable**: Có thể dừng và tiếp tục bằng cách comment out các chunk đã xử lý
4. **Parallel**: Hiện tại chạy tuần tự, có thể modify để chạy song song nhiều chunk

## Troubleshooting

### Out of Memory
- Giảm `batch_size` xuống 2 hoặc 4
- Sử dụng CPU thay vì GPU

### Network errors
- Kiểm tra HF_TOKEN
- Kiểm tra kết nối internet
- Script sẽ tự retry

### Model loading errors
- Cập nhật transformers: `pip install transformers>=4.51.0`
- Xóa cache: `rm -rf ~/.cache/huggingface/` 