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
#include <unordered_map>
#include <variant>
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

// A property value read from a ViewModel after an advance cycle.
using ViewModelPropertyValue = std::variant<std::monostate, // unset / not found
                                            bool,
                                            float,
                                            std::string>;

// Snapshot of watched ViewModel property values, produced by the background
// thread after each advance and readable by the render thread.
using ViewModelSnapshot = std::unordered_map<std::string, ViewModelPropertyValue>;

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
        // constructor to avoid a flash of empty content. Bindings whose render
        // callback must run only on the worker thread should set this false.
        bool runFirstFrameSync = true;

        // Self-paced render loop interval. When > 0, the worker computes `dt`
        // from a steady_clock between cycles and waits at most this many
        // microseconds on the wake CV; when 0 (legacy), the worker waits up
        // to 100 ms for `postElapsedTime` and uses the accumulated time as
        // `dt`. Setting this to e.g. 16667 (60 Hz) decouples the bg render
        // rate from the UI ticker rate, so Flutter's compositor wake (via
        // SurfaceProducer.scheduleFrame from the render-success callback)
        // can run at the target rate even when nothing on the UI thread is
        // calling postElapsedTime. Bindings that already drive the worker
        // from a UI ticker should leave this at 0.
        int targetFrameIntervalUs = 0;

        std::function<void(const std::string&)> logWarning;

        // Optional binding-owned flag that ThreadedScene writes to before
        // each render callback. Set to true iff the cycle published a new
        // event OR the new snapshot differs from the previous one. Lets
        // bindings gate push notifications (Dart_PostInteger_DL on
        // Android) on Dart-visible work, suppressing wakes on quiet
        // cycles where the worker animated but produced no diff.
        // The pointer must outlive the ThreadedScene (typically a member
        // of the owning binding).
        std::atomic<bool>* externalCycleOutputFlag = nullptr;
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

    // Register ViewModel properties to snapshot after each advance.
    // Call before or after construction; takes effect on the next advance.
    // Each name is a ViewModel property path (e.g., "currentAction").
    // The type is inferred from the ViewModel at snapshot time.
    void watchViewModelProperty(const std::string& propertyName);
    void unwatchViewModelProperty(const std::string& propertyName);

    // Read the latest snapshot of watched ViewModel property values.
    // Returns the snapshot produced after the most recent advance.
    // Protected by the same mutex as acquireCachedImage — zero extra
    // synchronization cost when called in the same frame.
    ViewModelSnapshot acquireViewModelSnapshot();

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

    // Atomically acquire the latest snapshot AND drain queued output events
    // under the same cached-image mutex acquisition. Eliminates the temporal
    // gap where separate acquireViewModelSnapshot() and pollReportedEvents()
    // calls can straddle a bg cycle (events from cycle N visible alongside
    // snapshot from cycle N-1).
    void acquireFrame(ViewModelSnapshot& outSnapshot,
                      std::vector<ThreadedOutputEvent>& outEvents);

    // Current dimensions.
    int width() const { return m_width.load(std::memory_order_relaxed); }
    int height() const { return m_height.load(std::memory_order_relaxed); }

    // Stop the background thread. Safe to call multiple times.
    // Called automatically by the destructor.
    void stop();

    bool isRunning() const
    {
        return m_running.load(std::memory_order_acquire);
    }

    // Note: there is no scene-level fatal-error flag. Fatal-error
    // reporting flows through the render callback's return value:
    // callers (e.g. the Android binding) set their own atomic before
    // returning nullptr on EGL/GL failure and surface that state to
    // application code. The scene only stops via stop() / destruction.

    // Total bg-thread cycles completed (state-machine advance + snapshot +
    // event collection + optional render callback). Bumped once per
    // runOneFrame, including cycles where the render callback was skipped
    // (e.g. zero-size surface, no callback registered) or returned nullptr.
    uint64_t advanceCount() const
    {
        return m_advanceCount.load(std::memory_order_relaxed);
    }

    // Total bg-thread cycles that produced a new RenderImage (render
    // callback ran and returned non-null). Diverges from advanceCount when
    // the bg thread advances state without producing a new frame.
    uint64_t renderedCount() const
    {
        return m_renderedCount.load(std::memory_order_relaxed);
    }

    // True if the most recently completed runOneFrame swap published either
    // new reported events OR a snapshot whose contents differed from the
    // previous cycle. Bindings can gate Dart-side push notifications on
    // this so quiet cycles (worker animating but no Dart-visible state
    // change) don't trigger spurious UI-thread wakes.
    //
    // Set under m_cachedImageMutex right after the snapshot/event swap;
    // read with acquire ordering so the render callback (which executes on
    // the bg thread inside the same runOneFrame, after the swap) sees the
    // correct value for the cycle that just completed.
    bool lastCycleProducedOutput() const
    {
        return m_lastCycleProducedOutput.load(std::memory_order_acquire);
    }

private:
    void threadMain();
    void pushEvent(ThreadedInputEvent event);
    bool applyInputEvents();
    void collectReportedEvents();
    void snapshotViewModelProperties();
    void runOneFrame(float dt);

    std::unique_ptr<ArtboardInstance> m_artboard;
    std::unique_ptr<StateMachineInstance> m_stateMachine;
    RenderCallback m_renderCallback;

    // Cached image, ViewModel snapshot, and ready-event buffer: written by
    // background thread, read by render thread. All three swap under the same
    // mutex at end of runOneFrame so a single acquireFrame() observes a
    // coherent snapshot/event pair from the same bg cycle.
    std::mutex m_cachedImageMutex;
    rcp<RenderImage> m_cachedImage;
    ViewModelSnapshot m_viewModelSnapshot;
    std::vector<ThreadedOutputEvent> m_readyEvents;

    // Watch list: written by render thread, read by background thread.
    std::mutex m_watchListMutex;
    std::vector<std::string> m_watchedProperties;

    // Staging areas (background thread only, no lock needed). Moved into the
    // ready slots under m_cachedImageMutex at end of cycle.
    ViewModelSnapshot m_pendingSnapshot;
    std::vector<ThreadedOutputEvent> m_pendingEvents;

    // Accumulated elapsed time.
    std::atomic<float> m_accumulatedTime{0.0f};

    // Input event queue (UI thread → bg thread). Output events are staged
    // and swapped together with the snapshot for coherent acquireFrame reads.
    ThreadedEventQueue<ThreadedInputEvent> m_inputQueue;

    // Thread lifecycle.
    std::thread m_thread;
    std::atomic<bool> m_running{false};
    std::atomic<uint64_t> m_advanceCount{0};
    std::atomic<uint64_t> m_renderedCount{0};
    std::atomic<bool> m_lastCycleProducedOutput{false};
    std::mutex m_wakeMutex;
    std::condition_variable m_wakeCV;
    bool m_wakeFlag = false; // protected by m_wakeMutex

    // Self-paced loop config. When `m_targetFrameIntervalUs > 0`, threadMain
    // ignores `m_accumulatedTime` and computes `dt` from a steady_clock
    // anchored at `m_lastTickClockUs` (set in the constructor and updated
    // every cycle). When 0 (legacy), the worker waits on `m_wakeFlag` for up
    // to 100 ms and uses the externally-posted accumulated time.
    int m_targetFrameIntervalUs = 0;
    int64_t m_lastTickClockUs = 0;

    // ViewModel instance for property lookups (accessed only on bg thread).
    rcp<ViewModelInstanceRuntime> m_viewModelInstance;
    std::function<void(const std::string&)> m_logWarning;
    std::atomic<bool>* m_externalCycleOutputFlag = nullptr;

    // Dimensions (atomics for lock-free reads from render thread).
    std::atomic<int> m_width{0};
    std::atomic<int> m_height{0};
};

} // namespace rive

#endif
