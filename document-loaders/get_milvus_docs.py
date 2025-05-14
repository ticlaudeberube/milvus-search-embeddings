import wget, zipfile

url='https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip'

def download_and_extract_docs(url=url, 
                            zip_file='milvus_docs_2.4.x_en.zip',
                            extract_dir='./milvus_docs'):
    wget.download(url)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

download_and_extract_docs()