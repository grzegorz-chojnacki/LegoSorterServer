import pika
import logging
import threading
from asyncio import Future


# Workaround for unhelpful thread exceptions
import sys
run_old = threading.Thread.run
def run(*args, **kwargs):
    try:
        run_old(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        sys.excepthook(*sys.exc_info())
threading.Thread.run = run


logging.getLogger('pika').setLevel(logging.FATAL)

class QueueService():
    def __init__(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = connection.channel()
        self.channel.queue_declare(queue='detect')
        self.channel.queue_declare(queue='classify')

    def start(self):
        thread = threading.Thread(target=lambda: self.channel.start_consuming())
        thread.start()

    def subscribe(self, queue_name: str, callback):
        logging.info(f'[QueueService] subscribe: {queue_name}')
        return self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=True)

    def publish(self, queue_name, body):
        logging.info(f'[QueueService] publish: {queue_name}')
        return self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body)

    async def rpc(self, queue_name, body):
        logging.info(f'[QueueService] rpc: {queue_name}')
        callback_queue = self.channel.queue_declare(
            queue='',
            exclusive=True).method.queue

        result = Future()

        def callback_function(channel, method, properties, body):
            result.set_result(body)

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            properties=pika.BasicProperties(reply_to=callback_queue),
            body=body)

        self.channel.basic_consume(
            queue=callback_queue,
            on_message_callback=callback_function,
            auto_ack=True)

        self.channel.connection.process_data_events(time_limit=None)
        return await result
