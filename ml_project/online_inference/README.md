Команды сборки образа
docker build -t alexcei64/online_inference:v1 .  
docker run -p 8000:8000 alexcei64/online_inference:v1  

Команды поднять локально образ  
docker pull alexcei64/online_inference:v1  
docker run -p 8000:8000 alexcei64/online_inference:v1  
python make_request.py (запускать в новом окружении)  
Дополнительно endpoint \predict_from_file\ - из файла  

PS  
ml_project/online_inference/requirements.txt - версия для Linux  
ml_project/requirements.txt - версия под Windows
