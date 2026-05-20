/*
 * Copyright 2026 Rive
 */

#include "rive/threaded_scene.hpp"
#include "rive/animation/state_machine_instance.hpp"
#include "rive/animation/state_machine_input_instance.hpp"
#include "rive/artboard.hpp"
#include "rive/event.hpp"
#include "rive/event_report.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_enum_runtime.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_number_runtime.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_boolean_runtime.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_string_runtime.hpp"
#include "rive/viewmodel/runtime/viewmodel_instance_trigger_runtime.hpp"

#include <chrono>

namespace rive
{

ThreadedScene::ThreadedScene(
    std::unique_ptr<ArtboardInstance> artboard,
    std::unique_ptr<StateMachineInstance> stateMachine,
    Config config,
    RenderCallback renderCallback,
    rcp<ViewModelInstanceRuntime> viewModelInstance) :
    m_artboard(std::move(artboard)),
    m_stateMachine(std::move(stateMachine)),
    m_renderCallback(std::move(renderCallback)),
    m_viewModelInstance(std::move(viewModelInstance)),
    m_logWarning(std::move(config.logWarning)),
    m_width(config.width),
    m_height(config.height),
    m_targetFrameIntervalUs(config.targetFrameIntervalUs)
{
    if (config.runFirstFrameSync)
    {
        runOneFrame(0.0f);
    }

    // Anchor the self-paced loop's dt clock right before the thread starts so
    // the first cycle's dt is the actual elapsed since construction, not a
    // stale or zero value. Harmless when targetFrameIntervalUs == 0.
    m_lastTickClockUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();

    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&ThreadedScene::threadMain, this);
}

ThreadedScene::~ThreadedScene()
{
    stop();
    if (m_stateMachine != nullptr && m_artboard != nullptr)
    {
#ifdef WITH_RIVE_TOOLS
        if (!m_stateMachine->hasExternalFocusManager())
        {
            auto* fm = m_stateMachine->internalFocusManager();
            if (fm != nullptr)
            {
                fm->setFocusChangedCallback(nullptr);
            }
        }
#endif
        m_artboard->cleanupFocusTree();
    }
}

void ThreadedScene::stop()
{
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
        m_running.store(false, std::memory_order_release);
        m_wakeFlag = true;
    }
    m_wakeCV.notify_one();
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

// --- Render thread API ---

void ThreadedScene::postElapsedTime(float seconds)
{
    if (seconds <= 0.0f)
    {
        return;
    }
    // Atomically accumulate time. The background thread will exchange this
    // to zero when it wakes up.
    float prev = m_accumulatedTime.load(std::memory_order_relaxed);
    while (!m_accumulatedTime.compare_exchange_weak(
        prev,
        prev + seconds,
        std::memory_order_relaxed))
    {
    }
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
        m_wakeFlag = true;
    }
    m_wakeCV.notify_one();
}

void ThreadedScene::pushEvent(ThreadedInputEvent event)
{
    m_inputQueue.push(std::move(event));
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
        m_wakeFlag = true;
    }
    m_wakeCV.notify_one();
}

void ThreadedScene::pointerDown(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerDown;
    event.position = position;
    event.pointerId = pointerId;
    pushEvent(std::move(event));
}

void ThreadedScene::pointerMove(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerMove;
    event.position = position;
    event.pointerId = pointerId;
    pushEvent(std::move(event));
}

void ThreadedScene::pointerUp(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerUp;
    event.position = position;
    event.pointerId = pointerId;
    pushEvent(std::move(event));
}

void ThreadedScene::pointerExit(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerExit;
    event.position = position;
    event.pointerId = pointerId;
    pushEvent(std::move(event));
}

void ThreadedScene::setBoolInput(const std::string& name, bool value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setBool;
    event.inputName = name;
    event.boolValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::setNumberInput(const std::string& name, float value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setNumber;
    event.inputName = name;
    event.floatValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::fireTrigger(const std::string& name)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::fireTrigger;
    event.inputName = name;
    pushEvent(std::move(event));
}

void ThreadedScene::setViewModelEnum(const std::string& propertyName,
                                     const std::string& value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelEnum;
    event.inputName = propertyName;
    event.stringValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::setViewModelNumber(const std::string& propertyName,
                                       float value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelNumber;
    event.inputName = propertyName;
    event.floatValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::setViewModelBool(const std::string& propertyName,
                                     bool value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelBool;
    event.inputName = propertyName;
    event.boolValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::setViewModelString(const std::string& propertyName,
                                       const std::string& value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelString;
    event.inputName = propertyName;
    event.stringValue = value;
    pushEvent(std::move(event));
}

void ThreadedScene::fireViewModelTrigger(const std::string& propertyName)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::fireViewModelTrigger;
    event.inputName = propertyName;
    pushEvent(std::move(event));
}

void ThreadedScene::watchViewModelProperty(const std::string& propertyName)
{
    std::lock_guard<std::mutex> lock(m_watchListMutex);
    for (const auto& name : m_watchedProperties)
    {
        if (name == propertyName)
        {
            return; // already watching
        }
    }
    m_watchedProperties.push_back(propertyName);
}

void ThreadedScene::unwatchViewModelProperty(const std::string& propertyName)
{
    std::lock_guard<std::mutex> lock(m_watchListMutex);
    for (auto it = m_watchedProperties.begin(); it != m_watchedProperties.end();
         ++it)
    {
        if (*it == propertyName)
        {
            m_watchedProperties.erase(it);
            return;
        }
    }
}

ViewModelSnapshot ThreadedScene::acquireViewModelSnapshot()
{
    std::lock_guard<std::mutex> lock(m_cachedImageMutex);
    return m_viewModelSnapshot;
}

void ThreadedScene::resize(int width, int height)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::resize;
    event.intValue = width;
    event.intValue2 = height;
    pushEvent(std::move(event));
}

rcp<RenderImage> ThreadedScene::acquireCachedImage()
{
    std::lock_guard<std::mutex> lock(m_cachedImageMutex);
    return m_cachedImage;
}

void ThreadedScene::pollReportedEvents(std::vector<ThreadedOutputEvent>& out)
{
    std::lock_guard<std::mutex> lock(m_cachedImageMutex);
    out.insert(out.end(),
               std::make_move_iterator(m_readyEvents.begin()),
               std::make_move_iterator(m_readyEvents.end()));
    m_readyEvents.clear();
}

void ThreadedScene::acquireFrame(ViewModelSnapshot& outSnapshot,
                                 std::vector<ThreadedOutputEvent>& outEvents)
{
    std::lock_guard<std::mutex> lock(m_cachedImageMutex);
    outSnapshot = m_viewModelSnapshot;
    outEvents.insert(outEvents.end(),
                     std::make_move_iterator(m_readyEvents.begin()),
                     std::make_move_iterator(m_readyEvents.end()));
    m_readyEvents.clear();
}

// --- Background thread ---

void ThreadedScene::threadMain()
{
    const bool selfPaced = (m_targetFrameIntervalUs > 0);
    while (m_running.load(std::memory_order_acquire))
    {
        {
            std::unique_lock<std::mutex> lock(m_wakeMutex);
            // Self-paced mode waits at most `m_targetFrameIntervalUs` so the
            // loop ticks at the configured cadence; legacy mode waits 100 ms
            // for an external `postElapsedTime` wake. Either mode also wakes
            // immediately on `m_wakeFlag` (input events) — except in
            // self-paced mode the predicate ignores `m_wakeFlag` so that
            // postElapsedTime can't short-circuit the interval and pin the
            // bg rate to the UI ticker rate. Input events still surface in
            // the next applyInputEvents at the interval boundary (≤ 16 ms
            // at 60 Hz — acceptable input latency for the threaded use case).
            const auto waitInterval =
                selfPaced
                    ? std::chrono::microseconds(m_targetFrameIntervalUs)
                    : std::chrono::microseconds(100 * 1000);
            m_wakeCV.wait_for(lock,
                              waitInterval,
                              [this, selfPaced] {
                                  if (selfPaced)
                                  {
                                      return !m_running.load(
                                          std::memory_order_acquire);
                                  }
                                  return m_wakeFlag ||
                                         !m_running.load(
                                             std::memory_order_acquire);
                              });
            m_wakeFlag = false;
        }

        if (!m_running.load(std::memory_order_acquire))
        {
            break;
        }

        bool hadEvents = applyInputEvents();
        float dt;
        if (selfPaced)
        {
            // Compute dt from steady_clock so the SM advance is independent
            // of how often `postElapsedTime` is called (or whether it is at
            // all — e.g. when an external Ticker is muted while the host
            // route is offscreen). Any `m_accumulatedTime` posted by the UI
            // side is folded in so paused/resumed transitions don't double-
            // count: legacy callers can still drive the SM during self-paced
            // operation without producing duplicate dt.
            const int64_t nowUs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            int64_t deltaUs = nowUs - m_lastTickClockUs;
            if (deltaUs < 0) deltaUs = 0;
            // Cap to ~250 ms so a long background pause (app suspended,
            // device thermal throttle) doesn't fire a giant dt that
            // teleports the SM forward on resume.
            constexpr int64_t maxDeltaUs = 250 * 1000;
            if (deltaUs > maxDeltaUs) deltaUs = maxDeltaUs;
            m_lastTickClockUs = nowUs;
            dt = static_cast<float>(deltaUs) / 1e6f +
                 m_accumulatedTime.exchange(0.0f, std::memory_order_relaxed);
            // Always advance + render in self-paced mode — even a 0-dt cycle
            // is useful because it produces a fresh GPU frame the compositor
            // can pick up after a SurfaceProducer.scheduleFrame.
            runOneFrame(dt);
        }
        else
        {
            dt = m_accumulatedTime.exchange(0.0f,
                                            std::memory_order_relaxed);
            if (dt == 0.0f && !hadEvents)
            {
                continue;
            }
            runOneFrame(dt);
        }
    }
}

bool ThreadedScene::applyInputEvents()
{
    thread_local std::vector<ThreadedInputEvent> drainBuffer;
    drainBuffer.clear();
    m_inputQueue.drainInto(drainBuffer);

    auto logMissing = [this](const char* kind, const std::string& name) {
        if (m_logWarning)
        {
            m_logWarning(std::string(kind) + ": no input named '" + name + "'");
        }
    };

    for (const auto& event : drainBuffer)
    {
        switch (event.type)
        {
            case ThreadedInputEvent::pointerDown:
                m_stateMachine->pointerDown(event.position, event.pointerId);
                break;
            case ThreadedInputEvent::pointerMove:
                m_stateMachine->pointerMove(event.position, 0, event.pointerId);
                break;
            case ThreadedInputEvent::pointerUp:
                m_stateMachine->pointerUp(event.position, event.pointerId);
                break;
            case ThreadedInputEvent::pointerExit:
                m_stateMachine->pointerExit(event.position, event.pointerId);
                break;
            case ThreadedInputEvent::setBool:
                if (auto* input =
                        m_stateMachine->getBool(event.inputName))
                {
                    input->value(event.boolValue);
                }
                else
                {
                    logMissing("setBool", event.inputName);
                }
                break;
            case ThreadedInputEvent::setNumber:
                if (auto* input =
                        m_stateMachine->getNumber(event.inputName))
                {
                    input->value(event.floatValue);
                }
                else
                {
                    logMissing("setNumber", event.inputName);
                }
                break;
            case ThreadedInputEvent::fireTrigger:
                if (auto* input =
                        m_stateMachine->getTrigger(event.inputName))
                {
                    input->fire();
                }
                else
                {
                    logMissing("fireTrigger", event.inputName);
                }
                break;
            case ThreadedInputEvent::resize:
                m_width.store(event.intValue, std::memory_order_relaxed);
                m_height.store(event.intValue2, std::memory_order_relaxed);
                break;
            case ThreadedInputEvent::setViewModelEnum:
                if (m_viewModelInstance)
                {
                    if (auto* prop =
                            m_viewModelInstance->propertyEnum(event.inputName))
                    {
                        prop->value(event.stringValue);
                    }
                    else
                    {
                        logMissing("setViewModelEnum[prop-null]", event.inputName);
                    }
                }
                else
                {
                    logMissing("setViewModelEnum[vmi-null]", event.inputName);
                }
                break;
            case ThreadedInputEvent::setViewModelNumber:
                if (m_viewModelInstance)
                {
                    if (auto* prop = m_viewModelInstance->propertyNumber(
                            event.inputName))
                    {
                        prop->value(event.floatValue);
                    }
                    else
                    {
                        logMissing("setViewModelNumber[prop-null]", event.inputName);
                    }
                }
                else
                {
                    logMissing("setViewModelNumber[vmi-null]", event.inputName);
                }
                break;
            case ThreadedInputEvent::setViewModelBool:
                if (m_viewModelInstance)
                {
                    if (auto* prop = m_viewModelInstance->propertyBoolean(
                            event.inputName))
                    {
                        prop->value(event.boolValue);
                    }
                    else
                    {
                        logMissing("setViewModelBool[prop-null]", event.inputName);
                    }
                }
                else
                {
                    logMissing("setViewModelBool[vmi-null]", event.inputName);
                }
                break;
            case ThreadedInputEvent::setViewModelString:
                if (m_viewModelInstance)
                {
                    if (auto* prop = m_viewModelInstance->propertyString(
                            event.inputName))
                    {
                        prop->value(event.stringValue);
                    }
                    else
                    {
                        logMissing("setViewModelString[prop-null]", event.inputName);
                    }
                }
                else
                {
                    logMissing("setViewModelString[vmi-null]", event.inputName);
                }
                break;
            case ThreadedInputEvent::fireViewModelTrigger:
                if (m_viewModelInstance)
                {
                    if (auto* prop = m_viewModelInstance->propertyTrigger(
                            event.inputName))
                    {
                        prop->trigger();
                    }
                    else
                    {
                        logMissing("fireViewModelTrigger[prop-null]", event.inputName);
                    }
                }
                else
                {
                    logMissing("fireViewModelTrigger[vmi-null]", event.inputName);
                }
                break;
        }
    }
    return !drainBuffer.empty();
}

void ThreadedScene::collectReportedEvents()
{
    for (size_t i = 0; i < m_stateMachine->reportedEventCount(); i++)
    {
        const EventReport report = m_stateMachine->reportedEventAt(i);
        ThreadedOutputEvent out;
        out.eventName = report.event()->name();
        out.secondsDelay = report.secondsDelay();
        m_pendingEvents.push_back(std::move(out));
    }
}

void ThreadedScene::snapshotViewModelProperties()
{
    if (!m_viewModelInstance)
    {
        return;
    }

    // Copy the watch list under its own lock.
    std::vector<std::string> watched;
    {
        std::lock_guard<std::mutex> lock(m_watchListMutex);
        watched = m_watchedProperties;
    }

    if (watched.empty())
    {
        return;
    }

    // Read current values on the background thread (safe — we own the VM).
    ViewModelSnapshot snapshot;
    for (const auto& name : watched)
    {
        if (auto* prop = m_viewModelInstance->propertyEnum(name))
        {
            snapshot[name] = prop->value();
        }
        else if (auto* prop = m_viewModelInstance->propertyNumber(name))
        {
            snapshot[name] = static_cast<float>(prop->value());
        }
        else if (auto* prop = m_viewModelInstance->propertyBoolean(name))
        {
            snapshot[name] = prop->value();
        }
        else if (auto* prop = m_viewModelInstance->propertyString(name))
        {
            snapshot[name] = prop->value();
        }
        else
        {
            snapshot[name] = std::monostate{};
        }
    }

    m_pendingSnapshot = std::move(snapshot);
}

void ThreadedScene::runOneFrame(float dt)
{
    // Fatal-error reporting flows through the render callback's return
    // value: callers (e.g. the Android binding) detect EGL/GL failure
    // inside the callback and set their own atomic before returning
    // nullptr. There is no try/catch here because every build that
    // consumes this code is compiled with -fno-exceptions (Flutter
    // Android, the unit_tests harness, etc.); a try/catch would compile
    // out and a thrown exception would call std::terminate anyway. If a
    // future build configuration enables exceptions, route fatal-error
    // reporting through return values rather than reintroducing a
    // catch-all here.
    m_stateMachine->advanceAndApply(dt);
    collectReportedEvents();
    snapshotViewModelProperties();

    int w = m_width.load(std::memory_order_relaxed);
    int h = m_height.load(std::memory_order_relaxed);

    rcp<RenderImage> newImage;
    if (m_renderCallback && w > 0 && h > 0)
    {
        newImage = m_renderCallback(m_artboard.get(), w, h);
    }

    // Swap cached image, ViewModel snapshot, and reported events
    // together under one lock so the render thread sees a coherent
    // triple from this bg cycle (event A and the snapshot reflecting
    // A's transition land atomically).
    {
        std::lock_guard<std::mutex> lock(m_cachedImageMutex);
        if (newImage)
        {
            m_cachedImage = std::move(newImage);
        }
        if (!m_pendingSnapshot.empty())
        {
            m_viewModelSnapshot = std::move(m_pendingSnapshot);
            // The standard only guarantees a moved-from unordered_map is
            // in a "valid but unspecified" state. Reset explicitly so the
            // next cycle's empty() check is portable.
            m_pendingSnapshot.clear();
        }
        if (!m_pendingEvents.empty())
        {
            m_readyEvents.insert(
                m_readyEvents.end(),
                std::make_move_iterator(m_pendingEvents.begin()),
                std::make_move_iterator(m_pendingEvents.end()));
            m_pendingEvents.clear();
        }
    }
}

} // namespace rive
