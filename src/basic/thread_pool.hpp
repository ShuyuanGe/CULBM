#pragma once

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <barrier>
#include "logger.hpp"
#include <functional>
#include <condition_variable>

namespace culbm::basic
{
    class ThreadPool
    {
        public:
            using Task = std::function<void(std::uint32_t, Logger&, std::barrier<>&)>;
        private:
            std::mutex                  mutex;
            std::queue<Task>            tasks;
            std::condition_variable     taskReady;
            std::condition_variable     tasksDone;
            std::barrier<>              barrier;
            std::vector<std::jthread>   workers;

            void workerLoop(std::stop_token stoken, std::uint32_t tid);
        public:
            ThreadPool(std::uint32_t numWorker);
            ~ThreadPool() noexcept;
            void addTask(Task task);
            void waitAll();
    };
}