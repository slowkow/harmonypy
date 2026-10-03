// harmonypy - A data alignment algorithm.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// A minimal fork-join thread pool built on std::thread, so the extension
// needs no OpenMP runtime.

#ifndef HARMONY_THREAD_POOL_HPP
#define HARMONY_THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace harmony {

// Runs tasks 0..n-1 on the calling thread plus n_threads - 1 workers.
// Threads claim tasks dynamically, so a task must not depend on which thread
// runs it. Callers combine per-task results in task order, which keeps every
// result identical for any number of threads.
class ThreadPool {
public:
    explicit ThreadPool(unsigned n_threads) : n_threads_(n_threads < 1 ? 1 : n_threads) {
        workers_.reserve(n_threads_ - 1);
        for (unsigned i = 1; i < n_threads_; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const { return n_threads_; }

    // Call fn(i) for every i in [0, n_tasks) and return when all calls have
    // finished. The first exception thrown by a task is rethrown here.
    void parallel_for(size_t n_tasks, const std::function<void(size_t)>& fn) {
        if (n_tasks == 0) return;
        if (workers_.empty() || n_tasks == 1) {
            for (size_t i = 0; i < n_tasks; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            n_tasks_ = n_tasks;
            next_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            busy_ = static_cast<unsigned>(workers_.size());
            ++generation_;
        }
        start_cv_.notify_all();
        run_tasks();
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this] { return busy_ == 0; });
            fn_ = nullptr;
            error = error_;
        }
        if (error) std::rethrow_exception(error);
    }

private:
    void run_tasks() {
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_tasks_) return;
            try {
                (*fn_)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
                // Skip the remaining tasks.
                next_.store(n_tasks_, std::memory_order_relaxed);
            }
        }
    }

    void worker_loop() {
        unsigned long long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            run_tasks();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--busy_ == 0) done_cv_.notify_one();
            }
        }
    }

    unsigned n_threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    const std::function<void(size_t)>* fn_ = nullptr;
    size_t n_tasks_ = 0;
    std::atomic<size_t> next_{0};
    std::exception_ptr error_;
    unsigned busy_ = 0;
    unsigned long long generation_ = 0;
    bool stop_ = false;
};

} // namespace harmony

#endif // HARMONY_THREAD_POOL_HPP
