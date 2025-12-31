#include <format>
#include "thread_pool.hpp"

namespace culbm::basic
{
    void ThreadPool::_workerLoop(std::stop_token stoken, std::uint32_t tid)
    {
        Logger logger(std::format("Worker-{}", tid));

        while(not stoken.stop_requested())
        {
            Task currTask;

            {
                std::unique_lock lock(_mtx);

                _waitNewTask.wait(lock, [&,this](){
                    return stoken.stop_requested() or (not _tasks.empty());
                });

                if(stoken.stop_requested()) break;

                currTask = _tasks.front();
            }

            try
            {
                currTask(tid, logger, _workerLoopBarrier);
            }
            catch(const std::exception& e)
            {
                logger.error(e.what());
            }

            _workerLoopBarrier.arrive_and_wait();

            if(tid == 0)
            {
                std::unique_lock lock (_mtx);
                _tasks.pop();
                if(_tasks.empty()) _finishAllTask.notify_all();
            }

            _workerLoopBarrier.arrive_and_wait();
        }
        logger.info("Exit successfully!");
    }

    ThreadPool::ThreadPool(std::uint32_t numWorker) : 
        _workerLoopBarrier(numWorker)
    {
        for(std::uint32_t tid=0 ; tid<numWorker ; ++tid)
        {
            _workers.emplace_back(&ThreadPool::_workerLoop, this, tid);
        }
    }

    ThreadPool::~ThreadPool() noexcept
    {
        waitAll();
        for(auto& worker : _workers) worker.request_stop();
        _waitNewTask.notify_all();
        for(auto& worker : _workers) worker.join();
    }

    void ThreadPool::addTask(Task task)
    {
        {
            std::unique_lock lock (_mtx);
            _tasks.emplace(std::move(task));
        }
        _waitNewTask.notify_all();
    }

    void ThreadPool::waitAll()
    {
        std::unique_lock lock (_mtx);

        _finishAllTask.wait(lock, [this]()->bool{
            return _tasks.empty();
        });
    }
}