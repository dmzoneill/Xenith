"""Unit tests for the plasma widget and the modules extracted from it"""

import importlib
import math
import subprocess  # nosec B404
import types
import unittest
from unittest import mock

from tests import gi_stubs

FAKES = gi_stubs.install()

# The fake gi modules must be installed before the widget modules are imported
easing = importlib.import_module("src.widgets.plasma_easing")
geometry = importlib.import_module("src.widgets.plasma_geometry")
palette = importlib.import_module("src.widgets.plasma_palette")
particles = importlib.import_module("src.widgets.plasma_particles")
renderer = importlib.import_module("src.widgets.plasma_renderer")
x11_window = importlib.import_module("src.widgets.x11_window")
plasma_widget = importlib.import_module("src.widgets.plasma_widget")


class EasingTest(unittest.TestCase):
    """plasma_easing"""

    def test_endpoints(self):
        """All easings map 0 to 0 and 1 to 1"""
        for func in (easing.ease_in_out_cubic, easing.ease_in_out_quintic, easing.ease_in_out_logarithmic):
            self.assertAlmostEqual(func(0.0), 0.0)
            self.assertAlmostEqual(func(1.0), 1.0)

    def test_values(self):
        """Known values match the original formulas"""
        self.assertAlmostEqual(easing.ease_in_out_cubic(0.25), 4 * 0.25**3)
        self.assertAlmostEqual(easing.ease_in_out_quintic(0.75), 1 - pow(-2 * 0.75 + 2, 5) / 2)
        self.assertAlmostEqual(easing.ease_in_out_logarithmic(0.25), 0.5 * math.log(1.5) / math.log(2))

    def test_logarithmic_clamps(self):
        """Out-of-range input is clamped"""
        self.assertAlmostEqual(easing.ease_in_out_logarithmic(-1), 0.0)
        self.assertAlmostEqual(easing.ease_in_out_logarithmic(2), 1.0)

    def test_lerp_color(self):
        """Interpolates each channel"""
        self.assertEqual(easing.lerp_color((0, 0, 0), (1, 2, 4), 0.5), (0.5, 1.0, 2.0))


class PaletteTest(unittest.TestCase):
    """plasma_palette"""

    def test_transition_lifecycle(self):
        """A transition starts at the displayed color and completes after its duration"""
        transition = palette.ColorTransition()
        self.assertFalse(transition.active)
        transition.start((1.0, 0.0, 0.0), now=10.0)
        self.assertTrue(transition.active)
        self.assertEqual(transition.displayed_color(), palette.STATE_BASE_COLORS["idle"])
        transition.update(11.0)
        self.assertAlmostEqual(transition.progress, 0.5)
        transition.update(12.5)
        self.assertEqual(transition.progress, 1.0)
        self.assertEqual(transition.current, (1.0, 0.0, 0.0))
        self.assertEqual(transition.displayed_color(), (1.0, 0.0, 0.0))

    def test_restart_mid_transition_captures_displayed_color(self):
        """Restarting mid-way starts from the currently displayed color (no flash)"""
        transition = palette.ColorTransition((0.0, 0.0, 0.0))
        transition.start((1.0, 1.0, 1.0), now=0.0)
        transition.update(1.0)
        shown = transition.displayed_color()
        transition.start((0.0, 1.0, 0.0), now=1.0)
        self.assertEqual(transition.current, shown)

    def test_tint(self):
        """Agent tint blends and clamps"""
        tint = palette.AgentTint(color_manager=mock.MagicMock())
        self.assertEqual(tint.apply((0.2, 0.4, 0.6)), (0.2, 0.4, 0.6))
        tint.set_color((1.0, 1.0, 1.0), 5)
        self.assertEqual(tint.intensity, 1.0)
        self.assertEqual(tint.apply((0.2, 0.4, 0.6)), (1.0, 1.0, 1.0))
        tint.color_manager.blend_agent_colors.return_value = (0.1, 0.2, 0.3, 0.4)
        tint.set_agents(["dev"])
        self.assertEqual((tint.color, tint.intensity, tint.active_agents), ((0.1, 0.2, 0.3), 0.4, ["dev"]))
        tint.set_agents([])
        self.assertEqual((tint.color, tint.intensity), (None, 0.0))

    def test_palette_unknown_state_uses_fallback(self):
        """Unknown states transition to the fallback grey"""
        pal = palette.PlasmaPalette()
        pal.transition_to_state("bogus", now=0.0)
        self.assertEqual(pal.transition.target, palette.FALLBACK_COLOR)


class ParticlesTest(unittest.TestCase):
    """plasma_particles"""

    def test_new_particle_ranges(self):
        """Random values are within their documented ranges"""
        particle = particles.new_particle()
        self.assertTrue(0.3 <= particle["distance"] <= 0.9)
        self.assertTrue(0.5 <= particle["speed"] <= 1.5)
        self.assertTrue(1 <= particle["size"] <= 3)

    def test_set_count(self):
        """Grows, shrinks and clamps the count"""
        system = particles.ParticleSystem(15)
        self.assertEqual(len(system.particles), 15)
        system.set_count(20)
        self.assertEqual(len(system.particles), 20)
        system.set_count(3)
        self.assertEqual(len(system.particles), 3)
        system.set_count(0)
        self.assertEqual((system.count, system.particles), (0, []))

    def test_advance_wraps(self):
        """Angle wraps past 2*pi"""
        particle = {"angle": 2 * math.pi, "speed": 1.0}
        particles.advance_particle(particle)
        self.assertAlmostEqual(particle["angle"], 0.01)

    def test_position_and_size(self):
        """Position stays within 90% of the radius"""
        particle = {"angle": 0.0, "speed": 1.0, "distance": 1.0, "size": 2.0, "phase": 0.0}
        self.assertEqual(particles.particle_position(particle, (10, 10), 10, 0.0), (19.0, 10.0))
        self.assertAlmostEqual(particles.particle_size(particle, 0.0), 2.0)


class GeometryTest(unittest.TestCase):
    """plasma_geometry"""

    def test_corner_position(self):
        """Each corner is offset from the area edges"""
        area = geometry.Area(100, 50, 1000, 800)
        cases = {
            "top-left": (120, 70),
            "top-right": (100 + 1000 - 120 - 20, 70),
            "bottom-left": (120, 50 + 800 - 120 - 20),
            "bottom-right": (960, 710),
        }
        for corner, expected in cases.items():
            self.assertEqual(geometry.corner_position(corner, area, (120, 120), 20), expected)
        self.assertIsNone(geometry.corner_position("middle", area, (120, 120), 20))

    def test_resize_helpers(self):
        """Resize corner detection and clamping"""
        self.assertTrue(geometry.in_resize_corner((110, 110), (120, 120)))
        self.assertFalse(geometry.in_resize_corner((10, 110), (120, 120)))
        self.assertEqual(geometry.dominant_delta(5, -10), -10)
        self.assertEqual(geometry.dominant_delta(-7, 3), -7)
        self.assertEqual(geometry.resized_size(120, 1000, (32, 500)), 500)
        self.assertEqual(geometry.resized_size(120, -1000, (32, 500)), 32)


class RendererTest(unittest.TestCase):
    """plasma_renderer"""

    def _frame(self, color=(1.0, 0.5, 0.25)):
        center, radius, max_radius = renderer.frame_geometry(120, 100, 10)
        return renderer.RenderFrame(mock.MagicMock(), center, radius, max_radius, color, 1.0)

    def test_frame_geometry(self):
        """Center and radii honour the margin"""
        self.assertEqual(renderer.frame_geometry(120, 100, 10), ((60, 50), 40 * 0.85, 40))

    def test_fill_radial_tints_stops(self):
        """Color stops are multiplied by the frame color"""
        frame = self._frame()
        cairo = FAKES["cairo"]
        cairo.RadialGradient.reset_mock()
        renderer.fill_radial(frame, 0, 10, [(0.0, 2, 2, 2, 0.5)])
        pattern = cairo.RadialGradient.return_value
        pattern.add_color_stop_rgba.assert_called_once_with(0.0, 2.0, 1.0, 0.5, 0.5)
        frame.cr.arc.assert_called_once_with(60, 50, 10, 0, 2 * math.pi)

    def test_draw_frame_each_state(self):
        """Every state draws its layers, particles and waves"""
        expected_fills = {"idle": 3, "listening": 3, "processing": 4, "responding": 3}
        for state, fills in expected_fills.items():
            frame = self._frame()
            renderer.draw_frame(frame, state, [particles.new_particle()])
            self.assertEqual(frame.cr.fill.call_count, fills + 1, state)
            frame.cr.paint.assert_called_once()

    def test_waves(self):
        """Listening draws one ring, idle none"""
        frame = self._frame()
        renderer.draw_waves(frame, "listening")
        self.assertEqual(frame.cr.stroke.call_count, 1)
        frame = self._frame()
        renderer.draw_waves(frame, "idle")
        frame.cr.stroke.assert_not_called()


class X11WindowTest(unittest.TestCase):
    """x11_window"""

    @mock.patch.object(x11_window.subprocess, "run")
    def test_tool_runners(self, run):
        """Returns False when the command cannot run"""
        self.assertTrue(x11_window.move_window(1, 2, 3))
        self.assertEqual(run.call_args.args[0], ["xdotool", "windowmove", "1", "2", "3"])
        self.assertFalse(run.call_args.kwargs["check"])
        self.assertTrue(x11_window.set_utility_type(4))
        self.assertEqual(run.call_args.args[0][:3], ["xprop", "-id", "4"])
        run.side_effect = FileNotFoundError
        self.assertFalse(x11_window.run_wmctrl("-l"))
        run.side_effect = subprocess.TimeoutExpired("x", 0.1)
        self.assertFalse(x11_window.run_xprop("-root"))

    @mock.patch.object(x11_window.subprocess, "run")
    def test_display_geometry(self, run):
        """Parses WIDTH HEIGHT and tolerates failures"""
        run.return_value = types.SimpleNamespace(returncode=0, stdout="1920 1080\n")
        self.assertEqual(x11_window.display_geometry(), geometry.Area(0, 0, 1920, 1080))
        run.return_value = types.SimpleNamespace(returncode=1, stdout="")
        self.assertIsNone(x11_window.display_geometry())
        run.return_value = types.SimpleNamespace(returncode=0, stdout="garbage x\n")
        self.assertIsNone(x11_window.display_geometry())

    @mock.patch.object(x11_window.subprocess, "run")
    def test_window_position(self, run):
        """Parses X= and Y= lines"""
        run.return_value = types.SimpleNamespace(returncode=0, stdout="WINDOW=1\nX=10\nY=20\nWIDTH=5\n")
        self.assertEqual(x11_window.window_position(1), {"X": 10, "Y": 20})
        run.return_value = types.SimpleNamespace(returncode=1, stdout="")
        self.assertEqual(x11_window.window_position(1), {})
        run.side_effect = OSError
        self.assertIsNone(x11_window.window_position(1))

    @mock.patch.object(x11_window.subprocess, "run")
    def test_set_window_above(self, run):
        """Falls back through wmctrl and xdotool only if xprop could run"""
        x11_window.set_window_above(7)
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ["xprop", "wmctrl", "xdotool"])
        run.reset_mock()
        run.side_effect = FileNotFoundError
        x11_window.set_window_above(7)
        self.assertEqual(run.call_count, 1)

    def test_spawn_quiet(self):
        """Async spawn reports GLib errors"""
        glib = FAKES["gi.repository.GLib"]
        glib.spawn_async.side_effect = None
        self.assertTrue(x11_window.move_window_async(1, 2, 3))
        self.assertEqual(glib.spawn_async.call_args.args[0], ["xdotool", "windowmove", "1", "2", "3"])
        glib.spawn_async.side_effect = gi_stubs.FakeGLibError
        self.assertFalse(x11_window.spawn_quiet(["x"]))
        glib.spawn_async.side_effect = None

    def test_get_xid_without_surface(self):
        """No surface means no xid"""
        self.assertIsNone(x11_window.get_xid(None))


@mock.patch.object(plasma_widget.x11_window, "display_geometry", return_value=None)
class PlasmaWidgetTest(unittest.TestCase):
    """PlasmaWidget behaviour with fake GTK"""

    def _widget(self):
        widget = plasma_widget.PlasmaWidget(mock.MagicMock())
        widget.drawing_area = mock.MagicMock()
        return widget

    def test_set_state_transitions_and_debounces(self, _geom):
        """State changes start a color transition; quick listening->idle is ignored"""
        widget = self._widget()
        widget.set_state("bogus")
        widget.set_state("listening")
        self.assertEqual(widget.get_opacity(), palette.STATE_OPACITIES["listening"])
        self.assertEqual(widget.state, "listening")
        widget.set_state("idle")
        self.assertEqual(widget.state, "listening")

    def test_set_opacity_clamps(self, _geom):
        """Opacity is clamped to 0-1"""
        widget = self._widget()
        widget.set_opacity(3)
        self.assertEqual(widget.get_opacity(), 1.0)

    def test_resize_drag(self, _geom):
        """Pressing in the corner then moving resizes within bounds"""
        widget = self._widget()
        widget.on_button_press(None, 1, 115, 115)
        widget.on_motion(None, 2000, 120)
        self.assertEqual(widget.default_size, (500, 500))
        widget.on_button_release(None, 1, 0, 0)
        widget.on_motion(None, 0, 0)
        self.assertEqual(widget.default_size, (500, 500))

    @mock.patch.object(plasma_widget.x11_window, "move_window")
    @mock.patch.object(plasma_widget.x11_window, "get_xid", return_value=5)
    def test_update_position_uses_monitor_fallback(self, _xid, move_window, _geom):
        """Falls back to the GTK monitor geometry when xdotool is unavailable"""
        widget = self._widget()
        monitor = mock.MagicMock()
        monitor.get_geometry.return_value = types.SimpleNamespace(x=0, y=0, width=1000, height=800)
        widget.display = mock.MagicMock()
        widget.display.get_primary_monitor.return_value = monitor
        widget.set_corner("top-right")
        move_window.assert_called_with(5, 860, 20)

    def test_animation_tick_reschedules(self, _geom):
        """Each tick schedules the next frame and returns False"""
        widget = self._widget()
        glib = FAKES["gi.repository.GLib"]
        widget.start_animation()
        tick = glib.timeout_add.call_args.args[1]
        glib.timeout_add.reset_mock()
        self.assertFalse(tick())
        self.assertEqual(glib.timeout_add.call_args.args[0], 33)
        widget.drawing_area.queue_draw.assert_called()


if __name__ == "__main__":
    unittest.main()
