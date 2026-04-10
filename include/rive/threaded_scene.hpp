/*
 * Copyright 2026 Rive
 */

#ifndef _RIVE_THREADED_SCENE_HPP_
#define _RIVE_THREADED_SCENE_HPP_

#include "rive/math/vec2d.hpp"
#include "rive/refcnt.hpp"
#include "rive/renderer.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_runtime.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rive
{
class ArtboardInstance;
class StateMachineInstance;

// A simple thread-safe queue for passing events between threads.
template <typename T> class ThreadedEventQueue
{
public:
    void push(T event)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(std::move(event));
    }

    bool tryPop(T& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
        {
            return false;
        }
        out = std::move(m_queue.front());
        m_queue.pop_front();
        return true;
    }

    void drainInto(std::vector<T>& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        out.insert(out.end(),
                   std::make_move_iterator(m_queue.begin()),
                   std::make_move_iterator(m_queue.end()));
        m_queue.clear();
    }

private:
    std::mutex m_mutex;
    std::deque<T> m_queue;
};

// Input event posted from the render/UI thread to the background thread.
struct ThreadedInputEvent
{
    enum Type
    {
        pointerDown,
        pointerMove,
        pointerUp,
        pointerExit,
        setBool,
        setNumber,
        fireTrigger,
        resize,
        setViewModelEnum,
        setViewModelNumber,
        setViewModelBool,
        setViewModelString,
        fireViewModelTrigger,
    };

    Type type;
    Vec2D position;
    int pointerId = 0;
    std::string inputName;
    std::string stringValue;
    float floatValue = 0.0f;
    bool boolValue = false;
    int intValue = 0;
    int intValue2 = 0;
};

// Output event forwarded from the background thread to the render/UI thread.
struct ThreadedOutputEvent
{
    std::string eventName;
    float secondsDelay = 0.0f;
};

// Runs advanceAndApply on a background thread and caches the rendered output
// as a RenderImage. The render thread blits this cached image each frame,
// decoupling the state machine evaluation rate from the UI frame rate.
//
// The background thread exclusively owns the ArtboardInstance and
// StateMachineInstance. The render thread never touches them.
//
// The caller provides a RenderCallback that handles the renderer-specific
// offscreen rendering (Skia SkSurface snapshot, GPU RenderCanvas, etc.).
class ThreadedScene
{
public:
    struct Config
    {
        int width = 0;
        int height = 0;
        // If true, the first advance+render runs synchronously in the
        // constructor to avoid a flash of empty content.
        bool runFirstFrameSync = true;
    };

    // Callback invoked on the background thread after each advanceAndApply.
    // Must render the artboard to an offscreen surface and return the result
    // as a RenderImage. The int parameters are the current width and height.
    using RenderCallback =
        std::function<rcp<RenderImage>(ArtboardInstance*, int width, int height)>;

    // Takes ownership of the artboard and state machine. They must not be
    // accessed after this call. The viewModelInstance is optional — pass
    // nullptr if the artboard doesn't use ViewModel data binding.
    ThreadedScene(std::unique_ptr<ArtboardInstance> artboard,
                  std::unique_ptr<StateMachineInstance> stateMachine,
                  Config config,
                  RenderCallback renderCallback,
                  rcp<ViewModelInstanceRuntime> viewModelInstance = nullptr);

    ~ThreadedScene();

    // Non-copyable, non-movable (owns a thread).
    ThreadedScene(const ThreadedScene&) = delete;
    ThreadedScene& operator=(const ThreadedScene&) = delete;

    // --- Render thread API (all methods are thread-safe) ---

    // Post elapsed time for the next advance cycle. Accumulates if the
    // background thread is slower than the caller.
    void postElapsedTime(float seconds);

    // Post pointer events.
    void pointerDown(Vec2D position, int pointerId = 0);
    void pointerMove(Vec2D position, int pointerId = 0);
    void pointerUp(Vec2D position, int pointerId = 0);
    void pointerExit(Vec2D position, int pointerId = 0);

    // Post state machine input changes (raw SM inputs by name).
    void setBoolInput(const std::string& name, bool value);
    void setNumberInput(const std::string& name, float value);
    void fireTrigger(const std::string& name);

    // Post ViewModel property changes (requires ViewModelInstanceRuntime).
    void setViewModelEnum(const std::string& propertyName,
                          const std::string& value);
    void setViewModelNumber(const std::string& propertyName, float value);
    void setViewModelBool(const std::string& propertyName, bool value);
    void setViewModelString(const std::string& propertyName,
                            const std::string& value);
    void fireViewModelTrigger(const std::string& propertyName);

    // Request a resize of the offscreen surface. Takes effect on the next
    // background thread cycle.
    void resize(int width, int height);

    // Get the latest cached image. Returns nullptr if no frame has been
    // produced yet. The returned rcp holds a ref, so the image stays alive
    // even if the background thread produces a new one.
    rcp<RenderImage> acquireCachedImage();

    // Drain reported events from the state machine (produced on the
    // background thread).
    void pollReportedEvents(std::vector<ThreadedOutputEvent>& out);

    // Current dimensions.
    int width() const { return m_width.load(std::memory_order_relaxed); }
    int height() const { return m_height.load(std::memory_order_relaxed); }

    // Stop the background thread. Safe to call multiple times.
    // Called automatically by the destructor.
    void stop();

    bool isRunning() const
    {
        return m_running.load(std::memory_order_relaxed);
    }

private:
    void threadMain();
    void pushEvent(ThreadedInputEvent event);
    void applyInputEvents();
    void collectReportedEvents();
    void runOneFrame(float dt);

    std::unique_ptr<ArtboardInstance> m_artboard;
    std::unique_ptr<StateMachineInstance> m_stateMachine;
    RenderCallback m_renderCallback;

    // Cached image: written by background thread, read by render thread.
    std::mutex m_cachedImageMutex;
    rcp<RenderImage> m_cachedImage;

    // Accumulated elapsed time.
    std::atomic<float> m_accumulatedTime{0.0f};

    // Event queues.
    ThreadedEventQueue<ThreadedInputEvent> m_inputQueue;
    ThreadedEventQueue<ThreadedOutputEvent> m_outputQueue;
    std::vector<ThreadedInputEvent> m_drainBuffer; // reused each cycle

    // Thread lifecycle.
    std::thread m_thread;
    std::atomic<bool> m_running{false};
    std::mutex m_wakeMutex;
    std::condition_variable m_wakeCV;

    // ViewModel instance for property lookups (accessed only on bg thread).
    rcp<ViewModelInstanceRuntime> m_viewModelInstance;

    // Dimensions (atomics for lock-free reads from render thread).
    std::atomic<int> m_width{0};
    std::atomic<int> m_height{0};
};

} // namespace rive

#endif
