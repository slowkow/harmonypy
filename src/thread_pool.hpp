// harmonypy - A data alignment algorithm.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// A minimal fork-join thread pool built on std::thread, so the extension
// needs no OpenMP runtime.

#ifndef HARMONY_THREAD_POOL_HPP
#define HARMONY_THREAD_POOL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace harmony {

// Runs tasks 0..n-1 on the calling thread plus up to n_threads - 1 workers.
// Threads claim tasks dynamically, so a task must not depend on which thread
// runs it, except to pick that thread's scratch space. Callers combine
// per-task results in task order, which keeps every result identical for any
// number of threads.
class ThreadPool {
public:
    // Starts n_threads - 1 workers. If the system refuses to start one (for
    // example a process or memory limit), the pool keeps the workers that
    // started; results do not depend on how many there are.
    explicit ThreadPool(unsigned n_threads) {
        const unsigned wanted = n_threads < 1 ? 1 : n_threads;
        workers_.reserve(wanted - 1);
        try {
            for (unsigned i = 1; i < wanted; ++i)
                workers_.emplace_back([this, i] { worker_loop(i); });
        } catch (...) {
            // std::system_error (or std::bad_alloc) from starting a thread.
        }
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

    // The number of threads that can run tasks, including the caller.
    unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }

    // Call fn(task, thread) for every task in [0, n_tasks) and return when all
    // calls have finished. thread is 0 for the calling thread and below size()
    // otherwise. At most n_tasks - 1 workers take part, and the caller waits
    // only for those. The first exception thrown by a task is rethrown here.
    void parallel_for(size_t n_tasks, const std::function<void(size_t, unsigned)>& fn) {
        if (n_tasks == 0) return;
        const unsigned helpers = static_cast<unsigned>(std::min<size_t>(workers_.size(), n_tasks - 1));
        if (helpers == 0) {
            for (size_t i = 0; i < n_tasks; ++i) fn(i, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            n_tasks_ = n_tasks;
            next_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            open_slots_ = helpers;
            ++generation_;
        }
        for (unsigned i = 0; i < helpers; ++i) start_cv_.notify_one();
        run_tasks(0);
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Every task has been claimed. Workers that have not joined yet
            // would find nothing to do, so stop admitting them and wait for
            // the ones that joined.
            open_slots_ = 0;
            done_cv_.wait(lock, [this] { return active_ == 0; });
            fn_ = nullptr;
            error = error_;
        }
        if (error) std::rethrow_exception(error);
    }

private:
    void run_tasks(unsigned thread) {
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_tasks_) return;
            try {
                (*fn_)(i, thread);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
                // Skip the remaining tasks.
                next_.store(n_tasks_, std::memory_order_relaxed);
            }
        }
    }

    void worker_loop(unsigned thread) {
        unsigned long long seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            start_cv_.wait(lock, [&] { return stop_ || (generation_ != seen && open_slots_ > 0); });
            if (stop_) return;
            seen = generation_;
            --open_slots_;
            ++active_;
            lock.unlock();
            run_tasks(thread);
            lock.lock();
            if (--active_ == 0) done_cv_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    const std::function<void(size_t, unsigned)>* fn_ = nullptr;
    size_t n_tasks_ = 0;
    std::atomic<size_t> next_{0};
    std::exception_ptr error_;
    unsigned open_slots_ = 0;   // workers that may still join the current call
    unsigned active_ = 0;       // workers running tasks of the current call
    unsigned long long generation_ = 0;
    bool stop_ = false;
};

// Runs one job on a separate thread; wait() or the destructor joins it.
// The job must not throw.
class BackgroundJob {
public:
    BackgroundJob() = default;
    ~BackgroundJob() { wait(); }
    BackgroundJob(const BackgroundJob&) = delete;
    BackgroundJob& operator=(const BackgroundJob&) = delete;

    // Start job on its own thread. Returns false, without running it, if the
    // system refuses to start a thread; the caller then runs the job itself.
    template <class F> bool try_start(F&& job) {
        wait();
        try {
            thread_ = std::thread(std::forward<F>(job));
        } catch (const std::system_error&) {
            return false;
        }
        return true;
    }
    void wait() {
        if (thread_.joinable()) thread_.join();
    }

private:
    std::thread thread_;
};

} // namespace harmony

#endif // HARMONY_THREAD_POOL_HPP
