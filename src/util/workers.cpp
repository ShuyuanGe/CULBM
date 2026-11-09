#include <exception>
#include <format>
#include "workers.hpp"

namespace culbm::util
{
    void Workers::worker_loop(std::stop_token stoken, std::uint32_t tid)
    {
        Logger logger(std::format("Worker-{}", tid));

        while (not stoken.stop_requested())
        {
            Task task;

            {
                std::unique_lock lock(mutex);

                task_ready.wait(lock, [&, this]() {
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
                std::terminate();
            }

            barrier.arrive_and_wait();

            if (tid == 0)
            {
                std::unique_lock lock (mutex);
                tasks.pop();
                if (tasks.empty()) tasks_done.notify_all();
            }

            barrier.arrive_and_wait();
        }
        logger.info("Exit successfully!");
    }

    Workers::Workers(std::uint32_t num_worker) :
        barrier(num_worker)
    {
        for (std::uint32_t tid = 0 ; tid < num_worker ; ++tid)
        {
            workers.emplace_back([this, tid](std::stop_token stoken) { worker_loop(stoken, tid); });
        }
    }

    Workers::~Workers() noexcept
    {
        wait();
        {
            std::lock_guard lock(mutex);
            for (auto& worker : workers) worker.request_stop();
        }
        task_ready.notify_all();
        for (auto& worker : workers) worker.join();
    }

    void Workers::submit(Task task)
    {
        {
            std::unique_lock lock (mutex);
            tasks.emplace(std::move(task));
        }
        task_ready.notify_all();
    }

    void Workers::wait()
    {
        std::unique_lock lock (mutex);

        tasks_done.wait(lock, [this]() -> bool {
            return tasks.empty();
        });
    }
}
