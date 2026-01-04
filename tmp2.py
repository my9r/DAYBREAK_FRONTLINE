import requests
from multiprocessing import Process
import time, os
import random as rd

header = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "cookie": "connect.sid=s%3A7xFCORpJB--CggMSFA_BOw43UbZxZyYy.XmP2BIvL8ysQKhxMgDA5EyaI4H9mumr4Tk8PlkEbZEw",
    "origin": "https://dingdingworld.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Google Chrome\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
}
E=10**20

def test_website_pressure(url, requests_per_process=10):
    for i in range(requests_per_process):
        try:
            response = requests.post(url,headers=header, json={"username": str(rd.randint(0,E)),"inviteCode": "findyourself", "password": "123456" * 11})
            print(f"Process {os.getpid()} - Request {i+1}: Status {response.status_code}")
        except Exception as e:
            print(f"Process {os.getpid()} - Request {i+1}: Error {e}")

if __name__ == "__main__":
    import os

    # url = input("Enter the URL to test: ").strip()
    url = "https://dingdingworld.com"
    # num_processes = int(input("Enter number of processes: ").strip())
    num_processes = 50
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
