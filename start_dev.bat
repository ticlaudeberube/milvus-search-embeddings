@echo off
cd /d "c:\Users\claud\Documents\workspace\milvus-search-embeddings"
call .venv\Scripts\activate.bat
echo Virtual environment activated
where python
python --version
cmd /k