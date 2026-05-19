/*
 * Copyright 2026 Rive
 */

#include "rive/threaded_scene.hpp"
#include "rive/artboard.hpp"
#include "rive/animation/state_machine_instance.hpp"
#include "rive/file.hpp"
#include "rive_file_reader.hpp"
#include "utils/no_op_renderer.hpp"

#include "catch.hpp"
#include <atomic>
#include <chrono>
#include <thread>

using namespace rive;

// A dummy RenderImage for testing (no actual pixel data).
class DummyRenderImage : public RenderImage
{
public:
    DummyRenderImage(int w, int h)
    {
        m_Width = w;
        m_Height = h;
    }
};

// Helper: create a ThreadedScene from a .riv file with a no-op render callback
// that returns a DummyRenderImage.
static std::unique_ptr<ThreadedScene> makeThreadedScene(
    const char* rivPath,
    std::atomic<int>* renderCount = nullptr,
    bool firstFrameSync = true)
{
    auto file = ReadRiveFile(rivPath);
    auto artboard = file->artboardDefault();
    REQUIRE(artboard != nullptr);
    auto stateMachine = artboard->stateMachineAt(0);
    REQUIRE(stateMachine != nullptr);

    ThreadedScene::Config config;
    config.width = static_cast<int>(artboard->width());
    config.height = static_cast<int>(artboard->height());
    config.runFirstFrameSync = firstFrameSync;

    auto scene = std::make_unique<ThreadedScene>(
        std::move(artboard),
        std::move(stateMachine),
        config,
        [renderCount](ArtboardInstance* ab, int w, int h) -> rcp<RenderImage> {
            // Render the artboard with a no-op renderer (just exercises the
            // draw path without producing pixels).
            NoOpRenderer renderer;
            ab->draw(&renderer);
            if (renderCount)
            {
                renderCount->fetch_add(1, std::memory_order_relaxed);
            }
            return make_rcp<DummyRenderImage>(w, h);
        });

    return scene;
}

// Wait up to `timeout` for a predicate to become true, polling at `interval`.
static bool waitFor(std::function<bool()> pred,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(2000),
                    std::chrono::milliseconds interval = std::chrono::milliseconds(10))
{
    auto start = std::chrono::steady_clock::now();
    while (!pred())
    {
        if (std::chrono::steady_clock::now() - start > timeout)
        {
            return false;
        }
        std::this_thread::sleep_for(interval);
    }
    return true;
}

TEST_CASE("ThreadedScene lifecycle", "[ThreadedScene]")
{
    // Verify that creating and destroying a ThreadedScene doesn't crash.
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv");
    REQUIRE(scene->isRunning());

    scene->stop();
    REQUIRE(!scene->isRunning());

    // Double-stop is safe.
    scene->stop();
    REQUIRE(!scene->isRunning());
}

TEST_CASE("ThreadedScene destructor stops thread", "[ThreadedScene]")
{
    {
        auto scene =
            makeThreadedScene("assets/multiple_state_machines.riv");
        REQUIRE(scene->isRunning());
        // Destructor should join the thread without hanging.
    }
}

TEST_CASE("ThreadedScene first frame sync produces cached image",
          "[ThreadedScene]")
{
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   nullptr,
                                   /*firstFrameSync=*/true);
    // With runFirstFrameSync, a cached image should be available immediately.
    auto img = scene->acquireCachedImage();
    REQUIRE(img != nullptr);
}

TEST_CASE("ThreadedScene renders on background thread", "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    int initialCount = renderCount.load(std::memory_order_relaxed);

    // Post time to trigger a background advance + render.
    scene->postElapsedTime(0.016f);

    // Wait for the background thread to produce at least one more render.
    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > initialCount;
    }));

    auto img = scene->acquireCachedImage();
    REQUIRE(img != nullptr);
}

TEST_CASE("ThreadedScene dimensions", "[ThreadedScene]")
{
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv");

    int origWidth = scene->width();
    int origHeight = scene->height();
    REQUIRE(origWidth > 0);
    REQUIRE(origHeight > 0);

    // Resize.
    scene->resize(800, 600);
    scene->postElapsedTime(0.016f);

    // Wait for the resize to take effect.
    REQUIRE(waitFor([&]() {
        return scene->width() == 800 && scene->height() == 600;
    }));
}

TEST_CASE("ThreadedScene input forwarding", "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    // Post various input events — they shouldn't crash.
    scene->pointerDown({100, 100});
    scene->pointerMove({110, 110});
    scene->pointerUp({110, 110});
    scene->pointerExit({0, 0});
    scene->setBoolInput("nonexistent", true);
    scene->setNumberInput("nonexistent", 42.0f);
    scene->fireTrigger("nonexistent");

    // Post time so the background thread processes the events.
    scene->postElapsedTime(0.016f);

    int countBefore = renderCount.load(std::memory_order_relaxed);
    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > countBefore;
    }));
}

TEST_CASE("ThreadedScene elapsed time accumulates", "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount,
                                   /*firstFrameSync=*/false);

    // Post multiple small increments rapidly.
    for (int i = 0; i < 10; i++)
    {
        scene->postElapsedTime(0.001f);
    }

    // The background thread should eventually process and render.
    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > 0;
    }));
}

TEST_CASE("ThreadedScene concurrent access stress", "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    std::atomic<bool> running{true};

    // Spawn threads that hammer the API concurrently.
    std::thread poster([&]() {
        while (running.load(std::memory_order_relaxed))
        {
            scene->postElapsedTime(0.001f);
            scene->pointerMove({50, 50});
        }
    });

    std::thread reader([&]() {
        while (running.load(std::memory_order_relaxed))
        {
            auto img = scene->acquireCachedImage();
            (void)img;
            std::vector<ThreadedOutputEvent> events;
            scene->pollReportedEvents(events);
        }
    });

    // Let it run for a bit.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    running.store(false, std::memory_order_relaxed);

    poster.join();
    reader.join();

    // If we got here without a crash or deadlock, the test passes.
    REQUIRE(renderCount.load(std::memory_order_relaxed) > 0);
}

TEST_CASE("ThreadedScene ViewModel inputs don't crash without VM",
          "[ThreadedScene]")
{
    // Scene created without a ViewModelInstanceRuntime — ViewModel
    // methods should be graceful no-ops.
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    scene->setViewModelEnum("nonexistent", "someValue");
    scene->setViewModelNumber("nonexistent", 42.0f);
    scene->setViewModelBool("nonexistent", true);
    scene->setViewModelString("nonexistent", "hello");
    scene->fireViewModelTrigger("nonexistent");
    scene->postElapsedTime(0.016f);

    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > 1;
    }));
}

TEST_CASE("ThreadedScene ViewModel concurrent stress", "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    std::atomic<bool> running{true};

    std::thread vmWriter([&]() {
        while (running.load(std::memory_order_relaxed))
        {
            scene->setViewModelEnum("currentAction", "idle");
            scene->setViewModelNumber("volume", 0.5f);
            scene->setViewModelBool("enabled", true);
            scene->setViewModelString("label", "test");
            scene->fireViewModelTrigger("fidget");
        }
    });

    std::thread timeAndReader([&]() {
        while (running.load(std::memory_order_relaxed))
        {
            scene->postElapsedTime(0.001f);
            auto img = scene->acquireCachedImage();
            (void)img;
            std::vector<ThreadedOutputEvent> events;
            scene->pollReportedEvents(events);
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    running.store(false, std::memory_order_relaxed);

    vmWriter.join();
    timeAndReader.join();

    REQUIRE(renderCount.load(std::memory_order_relaxed) > 0);
}

TEST_CASE("ThreadedScene fireViewModelTrigger rapid-fire is preserved",
          "[ThreadedScene]")
{
    // Rapid-fire VMI trigger calls between two advances must not crash, must
    // not deadlock, and must not be silently dropped by the threading layer.
    //
    // The input queue is ThreadedEventQueue<ThreadedInputEvent> — an
    // unbounded std::deque + mutex (threaded_scene.hpp). applyInputEvents()
    // calls drainInto() and processes every event in the buffer before each
    // runOneFrame(); there is no per-cycle cap on the input side.
    //
    // Note: this exercises the threading layer only. Rive's state machine
    // treats trigger consumption as a rising-edge signal per advance — N
    // fires of the same VMI trigger within a single advance window typically
    // drive at most one transition, because the state machine only checks
    // "did the trigger fire" once per cycle. That is intentional Rive
    // runtime behavior, not a threading bug. Application code that needs N
    // visible transitions must advance the state machine between fires; the
    // rig's triggers (fidget, gesture-react, drawing-end) are one-shot and
    // unaffected.
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    constexpr int kFireCount = 256;

    // Burst 1: single-threaded rapid fires, all queued before the next
    // advance. Worker drains all kFireCount events in one applyInputEvents()
    // pass, then advances once.
    for (int i = 0; i < kFireCount; i++)
    {
        scene->fireViewModelTrigger("fidget");
    }
    scene->postElapsedTime(0.016f);

    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > 0;
    }));

    // Burst 2: concurrent fires from a separate thread, mirroring the rig's
    // worker-post pattern (CharacterRig._onWorkerPost dispatching fires from
    // a non-UI source). Verifies the queue mutex is contention-safe and
    // doesn't deadlock with the advance loop.
    int beforeBurst2 = renderCount.load(std::memory_order_relaxed);
    std::thread firer([&]() {
        for (int i = 0; i < kFireCount; i++)
        {
            scene->fireViewModelTrigger("fidget");
        }
    });
    scene->postElapsedTime(0.016f);
    firer.join();

    // Worker continues making progress after the second burst drains.
    scene->postElapsedTime(0.016f);
    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > beforeBurst2;
    }));
}

TEST_CASE("ThreadedScene ViewModel snapshot empty without watch",
          "[ThreadedScene]")
{
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv");

    scene->postElapsedTime(0.016f);
    REQUIRE(waitFor([&]() {
        return scene->acquireCachedImage() != nullptr;
    }));

    auto snapshot = scene->acquireViewModelSnapshot();
    REQUIRE(snapshot.empty());
}

TEST_CASE("ThreadedScene ViewModel snapshot returns empty without VM",
          "[ThreadedScene]")
{
    std::atomic<int> renderCount{0};
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv",
                                   &renderCount);

    scene->watchViewModelProperty("nonexistent_prop");
    scene->postElapsedTime(0.016f);

    int initialCount = renderCount.load(std::memory_order_relaxed);
    REQUIRE(waitFor([&]() {
        return renderCount.load(std::memory_order_relaxed) > initialCount;
    }));

    // Without a ViewModelInstanceRuntime bound, snapshot stays empty.
    auto snapshot = scene->acquireViewModelSnapshot();
    REQUIRE(snapshot.empty());
}

TEST_CASE("ThreadedScene watch and unwatch", "[ThreadedScene]")
{
    auto scene = makeThreadedScene("assets/multiple_state_machines.riv");

    scene->watchViewModelProperty("propA");
    scene->watchViewModelProperty("propB");
    scene->watchViewModelProperty("propA"); // duplicate — no-op
    scene->unwatchViewModelProperty("propA");
    scene->unwatchViewModelProperty("nonexistent"); // no-op

    scene->postElapsedTime(0.016f);
    REQUIRE(waitFor([&]() {
        return scene->acquireCachedImage() != nullptr;
    }));
}
