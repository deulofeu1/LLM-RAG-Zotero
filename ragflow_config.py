import os
from ragflow_sdk import RAGFlow

# 1. 配置
API_KEY = "<your-api-key>"
BASE_URL = "http://<localhost>:9380"
DATASET_NAME = "<dataset-name>"
FOLDER_PATH = "<zotero-file-path>"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHUNK_METHOD = "paper"

# 2. 初始化 RAGFlow 客户端和数据集
def initialize_ragflow(dataset_name):
    try:
        rag_object = RAGFlow(api_key=API_KEY, base_url=BASE_URL)
        datasets = rag_object.list_datasets(name=dataset_name)
        if datasets:
            dataset = datasets[0]
            print(f"数据集 '{dataset_name}' 已存在，ID: {dataset.id}")
        else:
            dataset = rag_object.create_dataset(
                name=dataset_name,
                embedding_model=EMBEDDING_MODEL,
                chunk_method=CHUNK_METHOD,
            )
            print(f"数据集 '{dataset_name}' 创建成功，ID: {dataset.id}")
        return rag_object, dataset
    except Exception as e:
        print(f"初始化 RAGFlow 客户端或数据集时出错: {e}")
        return None, None


# 4. 获取 RAGFlow 上已上传的文件名列表
def get_uploaded_filenames_from_ragflow(dataset):
    uploaded_filenames = set()
    try:
        documents = dataset.list_documents(page_size=10000)  # 获取 RAGFlow 上的所有文档
        for doc in documents:
            uploaded_filenames.add(doc.name)  # 直接添加文件名
    except Exception as e:
        if "You don't own the document None" in str(e):
            print(
                f"警告：遇到权限问题，将跳过 RAGFlow 上的重复文件检查，仅依赖本地文件。错误信息: {e}"
            )
            return uploaded_filenames  # 返回一个空的 set，强制跳过 RAGFlow 检查
        else:
            print(
                f"警告：从 RAGFlow 获取已上传文件名时出错，错误信息: {e}。将跳过 RAGFlow 上的重复文件检查。"
            )
            return uploaded_filenames  # 返回一个空的 set，强制跳过 RAGFlow 检查

    return uploaded_filenames


# 5. 上传新文件的函数
def upload_new_files():
    print("开始检查和上传新文件...")

    # 初始化 RAGFlow 客户端和数据集
    rag_object, dataset = initialize_ragflow(DATASET_NAME)
    if not dataset:
        print("无法初始化 RAGFlow 客户端或数据集，跳过上传。")
        return

    for root, _, files in os.walk(FOLDER_PATH):  # 使用 os.walk 递归遍历文件夹
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)

                # 获取 RAGFlow 上已上传的文件名列表 (每次循环都获取)
                uploaded_filenames = get_uploaded_filenames_from_ragflow(dataset)

                # (a) 检查文件名是否已上传
                if filename in uploaded_filenames:
                    print(f"文件 '{filename}' 已存在于 RAGFlow，跳过上传")
                    continue

                # (b) 读取文件内容
                with open(file_path, "rb") as file:
                    blob = file.read()

                # (c) 上传文档
                try:
                    dataset.upload_documents([{"display_name": filename, "blob": blob}])
                    print(f"文件 '{filename}' 上传成功")

                    # (d) 异步解析文档 (重点：上传后立即解析)
                    documents = dataset.list_documents(page_size=10000)
                    documents = [doc for doc in documents if doc.name == filename]
                    if documents:
                        document_id = documents[0].id
                        dataset.async_parse_documents([document_id])
                        print(f"文件 '{filename}' 异步解析已启动")
                    else:
                        print(
                            f"文件 '{filename}' 上传后，未找到对应的文档记录, 跳过异步解析"
                        )

                except Exception as e:
                    print(f"上传文件 '{filename}' 失败: {e}")

    print("文件检查和上传完成。")
