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
    m_width(config.width),
    m_height(config.height)
{
    if (config.runFirstFrameSync)
    {
        runOneFrame(0.0f);
    }

    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&ThreadedScene::threadMain, this);
}

ThreadedScene::~ThreadedScene() { stop(); }

void ThreadedScene::stop()
{
    bool expected = true;
    if (!m_running.compare_exchange_strong(expected,
                                           false,
                                           std::memory_order_release,
                                           std::memory_order_relaxed))
    {
        return; // already stopped
    }
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
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
    while (m_running.load(std::memory_order_acquire))
    {
        {
            std::unique_lock<std::mutex> lock(m_wakeMutex);
            m_wakeCV.wait_for(lock,
                              std::chrono::milliseconds(100),
                              [this] {
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

        float dt = m_accumulatedTime.exchange(0.0f, std::memory_order_relaxed);
        applyInputEvents();

        if (dt == 0.0f && m_drainBuffer.empty())
        {
            continue;
        }

        runOneFrame(dt);
    }
}

void ThreadedScene::applyInputEvents()
{
    m_drainBuffer.clear();
    m_inputQueue.drainInto(m_drainBuffer);

    for (const auto& event : m_drainBuffer)
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
                break;
            case ThreadedInputEvent::setNumber:
                if (auto* input =
                        m_stateMachine->getNumber(event.inputName))
                {
                    input->value(event.floatValue);
                }
                break;
            case ThreadedInputEvent::fireTrigger:
                if (auto* input =
                        m_stateMachine->getTrigger(event.inputName))
                {
                    input->fire();
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
                }
                break;
        }
    }
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
    // The render callback executes user code on the background thread and can
    // throw (EGL/GL failures surfaced as exceptions, bad_alloc from rcp, etc.).
    // An uncaught exception out of threadMain would call std::terminate and
    // bypass the FFI fatal-error path, so contain it here and stop the loop.
    try
    {
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
    catch (...)
    {
        m_fatalError.store(true, std::memory_order_release);
        m_running.store(false, std::memory_order_release);
    }
}

} // namespace rive
