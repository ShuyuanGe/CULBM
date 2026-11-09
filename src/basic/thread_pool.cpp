#include <format>
#include "thread_pool.hpp"

namespace culbm::basic
{
    void ThreadPool::workerLoop(std::stop_token stoken, std::uint32_t tid)
    {
        Logger logger(std::format("Worker-{}", tid));

        while (not stoken.stop_requested())
        {
            Task task;

            {
                std::unique_lock lock(mutex);

                taskReady.wait(lock, [&, this]() {
                    return stoken.stop_requested() or (not tasks.empty());
                });

                if (stoken.stop_requested()) break;

                task = tasks.front();
            }

            try
            {
                task(tid, logger, barrier);
            }
            catch (const std::exception& e)
            {
                logger.error(e.what());
            }

            barrier.arrive_and_wait();

            if (tid == 0)
            {
                std::unique_lock lock (mutex);
                tasks.pop();
                if (tasks.empty()) tasksDone.notify_all();
            }

            barrier.arrive_and_wait();
        }
        logger.info("Exit successfully!");
    }

    ThreadPool::ThreadPool(std::uint32_t numWorker) : 
        barrier(numWorker)
    {
        for (std::uint32_t tid = 0 ; tid < numWorker ; ++tid)
        {
            workers.emplace_back([this, tid](std::stop_token stoken) { workerLoop(stoken, tid); });
        }
    }

    ThreadPool::~ThreadPool() noexcept
    {
        waitAll();
        {
            std::lock_guard lock(mutex);
            for (auto& worker : workers) worker.request_stop();
        }
        taskReady.notify_all();
        for (auto& worker : workers) worker.join();
    }

    void ThreadPool::addTask(Task task)
    {
        {
            std::unique_lock lock (mutex);
            tasks.emplace(std::move(task));
        }
        taskReady.notify_all();
    }

    void ThreadPool::waitAll()
    {
        std::unique_lock lock (mutex);

        tasksDone.wait(lock, [this]() -> bool {
            return tasks.empty();
        });
    }
}
