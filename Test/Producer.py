# coding=utf-8
# @Author:SDY
# @File:Producer.py
# @Time:2024/4/10 20:36
# @Introduction: 读取文件数据发送到kafka
import configparser
import sys

from kafka import KafkaProducer

def send_data_to_kafka(file_path,kafka_bootstrap_servers,topic_name):
    kafkaProducer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
    try:
        with open(file_path,'rb') as file:
            for line in file:
                kafkaProducer.send(topic_name,line.strip())
                print("data:{} sent successfully to Kafka topic: {}".format(topic_name,line.strip()))
    except Exception as e:
        print("Error:",e)
    finally:
        kafkaProducer.close()

if __name__ == '__main__':
    file_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('config.ini')
    # 从配置文件中获取 Kafka 相关的配置
    kafka_bootstrap_servers = config['kafka']['bootstrap_servers']
    kafka_topic = config['kafka']['topic']
    send_data_to_kafka(file_path,kafka_bootstrap_servers,kafka_topic)

