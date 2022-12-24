import sys
import aiohttp
import asyncio


def get_urls(file):
    """get urls"""
    with open(file, 'r') as filed:
        for line in filed.readlines():
            yield line


async def fetch_url(url, session, num):
    async with session.get(url) as resp:
        data = await resp.read()

        return resp.status, len(data), num


async def worker(queue, session, num):
    while True:
        url = await queue.get()

        try:
            res = await fetch_url(url, session, num)
            result.append(res)
        finally:
            queue.task_done()


async def fetch_batch_urls(queue, workers):
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(queue, session, i))
            for i in range(workers)
        ]
        await queue.join()

        for task in tasks:
            task.cancel()


async def main(FILE_PATH, WORKERS):
    urls_queue = asyncio.Queue()

    for url in get_urls(FILE_PATH):
        await urls_queue.put(url)

    await fetch_batch_urls(urls_queue, WORKERS)


if __name__ == '__main__':
    result = []
    # if len(sys.argv) == 3:
    #     N = int(sys.argv[1])
    #     file_path = sys.argv[2]
    #     asyncio.create_task(main(file_path, N))
    # elif len(sys.argv) == 4:
    #     N = int(sys.argv[2])
    #     file_path = sys.argv[3]
    #     asyncio.create_task(main(file_path, N))
    # else:
    #     print('Input "python fetcher.py -c 10 urls.txt" or "python fetcher.py 10 urls.txt"')

    # asyncio.create_task(main('urls.txt', 10))

    asyncio.run(main('urls.txt', 10))
