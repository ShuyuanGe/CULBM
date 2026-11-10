#pragma once

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <barrier>
#include <cstdint>
#include "logger.hpp"
#include <functional>
#include <condition_variable>

namespace culbm::util
{
    class Workers
    {
        public:
            using Task = std::function<void(std::uint32_t, Logger&, std::barrier<>&)>;
        private:
            std::mutex                  mutex;
            std::queue<Task>            tasks;
            std::condition_variable     task_ready;
            std::condition_variable     tasks_done;
            std::barrier<>              barrier;
            std::vector<std::jthread>   workers;

            void worker_loop(std::stop_token stoken, std::uint32_t tid);
        public:
            Workers(std::uint32_t num_worker);
            ~Workers() noexcept;
            void submit(Task task);
            void wait();
    };
}