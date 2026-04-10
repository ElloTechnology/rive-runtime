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

#include <algorithm>
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

    m_running.store(true, std::memory_order_relaxed);
    m_thread = std::thread(&ThreadedScene::threadMain, this);
}

ThreadedScene::~ThreadedScene() { stop(); }

void ThreadedScene::stop()
{
    bool expected = true;
    if (!m_running.compare_exchange_strong(expected, false))
    {
        return; // already stopped
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
    m_wakeCV.notify_one();
}

void ThreadedScene::pointerDown(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerDown;
    event.position = position;
    event.pointerId = pointerId;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::pointerMove(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerMove;
    event.position = position;
    event.pointerId = pointerId;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::pointerUp(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerUp;
    event.position = position;
    event.pointerId = pointerId;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::pointerExit(Vec2D position, int pointerId)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::pointerExit;
    event.position = position;
    event.pointerId = pointerId;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setBoolInput(const std::string& name, bool value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setBool;
    event.inputName = name;
    event.boolValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setNumberInput(const std::string& name, float value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setNumber;
    event.inputName = name;
    event.floatValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::fireTrigger(const std::string& name)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::fireTrigger;
    event.inputName = name;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setViewModelEnum(const std::string& propertyName,
                                     const std::string& value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelEnum;
    event.inputName = propertyName;
    event.stringValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setViewModelNumber(const std::string& propertyName,
                                       float value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelNumber;
    event.inputName = propertyName;
    event.floatValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setViewModelBool(const std::string& propertyName,
                                     bool value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelBool;
    event.inputName = propertyName;
    event.boolValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::setViewModelString(const std::string& propertyName,
                                       const std::string& value)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::setViewModelString;
    event.inputName = propertyName;
    event.stringValue = value;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::fireViewModelTrigger(const std::string& propertyName)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::fireViewModelTrigger;
    event.inputName = propertyName;
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

void ThreadedScene::resize(int width, int height)
{
    ThreadedInputEvent event;
    event.type = ThreadedInputEvent::resize;
    event.floatValue = static_cast<float>(width);
    event.floatValue2 = static_cast<float>(height);
    m_inputQueue.push(std::move(event));
    m_wakeCV.notify_one();
}

rcp<RenderImage> ThreadedScene::acquireCachedImage()
{
    std::lock_guard<std::mutex> lock(m_cachedImageMutex);
    return m_cachedImage;
}

void ThreadedScene::pollReportedEvents(std::vector<ThreadedOutputEvent>& out)
{
    m_outputQueue.drainInto(out);
}

// --- Background thread ---

void ThreadedScene::threadMain()
{
    while (m_running.load(std::memory_order_relaxed))
    {
        // Wait for work: elapsed time posted or input events queued.
        {
            std::unique_lock<std::mutex> lock(m_wakeMutex);
            m_wakeCV.wait_for(lock, std::chrono::milliseconds(100));
        }

        if (!m_running.load(std::memory_order_relaxed))
        {
            break;
        }

        // Consume accumulated elapsed time.
        float dt = m_accumulatedTime.exchange(0.0f, std::memory_order_relaxed);

        // Apply queued input events to the state machine.
        applyInputEvents();

        // Advance and render.
        runOneFrame(dt);
    }
}

void ThreadedScene::applyInputEvents()
{
    std::vector<ThreadedInputEvent> events;
    m_inputQueue.drainInto(events);

    for (const auto& event : events)
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
                m_width.store(static_cast<int>(event.floatValue),
                              std::memory_order_relaxed);
                m_height.store(static_cast<int>(event.floatValue2),
                               std::memory_order_relaxed);
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
        m_outputQueue.push(std::move(out));
    }
}

void ThreadedScene::runOneFrame(float dt)
{
    m_stateMachine->advanceAndApply(dt);
    collectReportedEvents();

    int w = m_width.load(std::memory_order_relaxed);
    int h = m_height.load(std::memory_order_relaxed);

    if (m_renderCallback && w > 0 && h > 0)
    {
        rcp<RenderImage> newImage =
            m_renderCallback(m_artboard.get(), w, h);
        if (newImage)
        {
            std::lock_guard<std::mutex> lock(m_cachedImageMutex);
            m_cachedImage = std::move(newImage);
        }
    }
}

} // namespace rive
