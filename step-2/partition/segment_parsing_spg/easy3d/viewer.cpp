/*
*	Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
*	https://3d.bk.tudelft.nl/liangliang/
*
*	This file is part of Easy3D: software for processing and rendering
*   meshes and point clouds.
*
*	Easy3D is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	Easy3D is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <easy3d/viewer.h>

#include <thread>
#include <chrono>
#include <iostream>


#if !defined(_WIN32)
#  include <unistd.h>
#  include <sys/wait.h>
#endif


#include <3rd_party/glew/include/GL/glew.h>		// Initialize with glewInit() 
#include <3rd_party/glfw/include/GLFW/glfw3.h>	// Include glfw3.h after our OpenGL definitions

#include <easy3d/surface_mesh.h>
#include <easy3d/point_cloud.h>
#include <easy3d/drawable.h>
#include <easy3d/shader_program.h>
#include <easy3d/resources.h>
#include <easy3d/dialogs.h>
#include <easy3d/transform.h>
#include <easy3d/camera.h>
#include <easy3d/manipulated_camera_frame.h>
#include <easy3d/file.h>
#include <easy3d/point_cloud_io.h>
#include <easy3d/surface_mesh_io.h>
#include <easy3d/ply_reader_writer.h>


namespace easy3d {


	Viewer::Viewer(
		const std::string& title /* = "easy3d Viewer" */,
		int samples /* = 4 */,
		int gl_major /* = 3 */,
		int gl_minor /* = 2 */,
		bool full_screen /* = false */,
		bool resizable /* = true */,
		int depth_bits /* = 24 */,
		int stencil_bits /* = 8 */
	)
        : title_(title)
        , camera_(nullptr)
        , samples_(0)
        , full_screen_(full_screen)
        , width_(1280)	// default width
        , height_(960)	// default height
        , process_events_(true)
        , pressed_key_(GLFW_KEY_UNKNOWN)
        , show_corner_axes_(true)
        , axes_(nullptr)
        , points_program_(nullptr)
        , lines_program_(nullptr)
        , surface_program_(nullptr)
        , model_idx_(-1)
	{
#if !defined(_WIN32)
		/* Avoid locale-related number parsing issues */
		setlocale(LC_NUMERIC, "C");
#endif

		glfwSetErrorCallback(
			[](int error, const char *descr) {
			if (error == GLFW_NOT_INITIALIZED)
				return; /* Ignore */
			std::cerr << "GLFW error " << error << ": " << descr << std::endl;
		});

		if (!glfwInit())
			throw std::runtime_error("Could not initialize GLFW!");

		glfwSetTime(0);

		// Reset the hints, allowing viewers to have different hints.
		glfwDefaultWindowHints();

		glfwWindowHint(GLFW_SAMPLES, samples);

		glfwWindowHint(GLFW_STENCIL_BITS, stencil_bits);
		glfwWindowHint(GLFW_DEPTH_BITS, depth_bits);

		/* Request a forward compatible OpenGL glMajor.glMinor core profile context.
		   Default value is an OpenGL 3.2 core profile context. */
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_minor);

#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);	// 3.2+ only
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// 3.0+ only
#else
		if (gl_major >= 3) {
			if (gl_minor >= 2)
				glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);	// 3.2+ only
			if (gl_minor >= 0)
				glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);			// 3.0+ only
		}
#endif

		// make the whole window transparent
		//glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

		glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
		glfwWindowHint(GLFW_RESIZABLE, resizable ? GL_TRUE : GL_FALSE);

		if (full_screen) {
			GLFWmonitor *monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode *mode = glfwGetVideoMode(monitor);
			window_ = glfwCreateWindow(mode->width, mode->height, title.c_str(), monitor, nullptr);
		}
		else {
            window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
		}
        glfwSetWindowUserPointer(window_, this);

		if (!window_) {
			glfwTerminate();
			throw std::runtime_error("Could not create an OpenGL " +
				std::to_string(gl_major) + "." + std::to_string(gl_minor) + " context!");
		}

		glfwMakeContextCurrent(window_);
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSwapInterval(1); // Enable vsync

		// Load OpenGL and its extensions
		if (glewInit() != GLEW_OK) {
			glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
			throw std::runtime_error("Failed to load OpenGL and its extensions!");
		}

#if 0
		std::cout << "OpenGL Version requested: " << gl_major << "." << gl_minor << std::endl;
		int major = glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR);
		int minor = glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR);
		int rev = glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION);
		std::cout << "OpenGL version received:  " << major << "." << minor << "." << rev << std::endl;
		std::cout << "Supported OpenGL:         " << glGetString(GL_VERSION) << std::endl;
		std::cout << "Supported GLSL:           " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
#endif

		std::string vender(reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
		if (vender.find("Intel") != std::string::npos) {
			std::cerr << "Detected Intel HD Graphics card, disabling MSAA as a precaution .." << std::endl;
		}

		glGetIntegerv(GL_SAMPLES, &samples_);
		// warn the user if the expected request was not satisfied
		if (samples > 0 && samples_ != samples) {
			if (samples_ == 0)
				printf("MSAA is not available with %i samples\n", samples);
			else {
				int max_num = 0;
				glGetIntegerv(GL_MAX_SAMPLES, &max_num);
				printf("MSAA is available with %i samples (%i requested, max support is %i)\n", samples_, samples, max_num);
			}
		}

        int fb_width, fb_height;
        glfwGetFramebufferSize(window_, &fb_width, &fb_height);
        int win_width, win_height;
        glfwGetWindowSize(window_, &win_width, &win_height);  // get the actual window size
        highdpi_ = static_cast<double>(fb_width)/win_width;

        glViewport(0, 0, fb_width, fb_height);
        glEnable(GL_DEPTH_TEST);

        camera_ = new Camera;
        camera_->setScreenWidthAndHeight(width_, height_);

		background_color_[0] = background_color_[1] = background_color_[2] = 0.3f;
		mouse_x_ = mouse_y_ = 0;
		mouse_pressed_x_ = mouse_pressed_y_ = 0;
		button_ = -1;
		pressed_key_ = GLFW_KEY_UNKNOWN;
		modifiers_ = 0;
		drag_active_ = false;
		process_events_ = true;

        setup_callbacks();

#if defined(__APPLE__)
		/* Poll for events once before starting a potentially lengthy loading process.*/
		glfwPollEvents();
#endif

        std::cout << usage();
	}


	void Viewer::setup_callbacks() {
		glfwSetCursorPosCallback(window_, [](GLFWwindow *win, double x, double y)
		{
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;

            int w, h;
            glfwGetWindowSize(win, &w, &h);
			if (x >= 0 && x <= w && y >= 0 && y <= h)
				viewer->callback_event_cursor_pos(x, y);
			else if (viewer->drag_active_) {
				// Restrict the cursor to be within the client area during dragging
				if (x < 0) x = 0;	if (x > w) x = w;
				if (y < 0) y = 0;	if (y > h) y = h;
				glfwSetCursorPos(win, x, y);
			}
		});

		glfwSetMouseButtonCallback(window_, [](GLFWwindow *win, int button, int action, int modifiers) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			viewer->callback_event_mouse_button(button, action, modifiers);
		});

		glfwSetKeyCallback(window_, [](GLFWwindow *win, int key, int scancode, int action, int mods) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			(void)scancode;
			viewer->callback_event_keyboard(key, action, mods);
		});

		glfwSetCharCallback(window_, [](GLFWwindow *win, unsigned int codepoint) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			viewer->callback_event_character(codepoint);
		});

		glfwSetDropCallback(window_, [](GLFWwindow *win, int count, const char **filenames) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			viewer->callback_event_drop(count, filenames);
		});

		glfwSetScrollCallback(window_, [](GLFWwindow *win, double dx, double dy) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			viewer->callback_event_scroll(dx, dy);
		});

        glfwSetWindowSizeCallback(window_, [](GLFWwindow *win, int width, int height) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			if (!viewer->process_events_)
				return;
			viewer->callback_event_resize(width, height);
		});

		// notify when the screen has lost focus (e.g. application switch)
		glfwSetWindowFocusCallback(window_, [](GLFWwindow *win, int focused) {
			Viewer* viewer = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(win));
			viewer->focus_event(focused != 0);// true for focused
		});

		glfwSetWindowCloseCallback(window_, [](GLFWwindow *win) {
			glfwSetWindowShouldClose(win, true);
		});
	}


	bool Viewer::callback_event_cursor_pos(double x, double y) {
        int px = static_cast<int>(x);
        int py = static_cast<int>(y);
		try {
			int dx = px - mouse_x_;
            int dy = py - mouse_y_;
			mouse_x_ = px;
			mouse_y_ = py;
			if (drag_active_)
				return mouse_drag_event(px, py, dx, dy, button_, modifiers_);
            else
				return mouse_free_move_event(px, py, dx, dy, modifiers_);
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what() << std::endl;
			return false;
		}
	}


	bool Viewer::callback_event_mouse_button(int button, int action, int modifiers) {
		try {
			if (action == GLFW_PRESS) {
				drag_active_ = true;
				button_ = button;
				modifiers_ = modifiers;
				mouse_pressed_x_ = mouse_x_;
				mouse_pressed_y_ = mouse_y_;
				return mouse_press_event(mouse_x_, mouse_y_, button, modifiers);
			}
			else if (action == GLFW_RELEASE) {
				drag_active_ = false;
				return mouse_release_event(mouse_x_, mouse_y_, button, modifiers);
			}
			else {
				drag_active_ = false;
				std::cout << "GLFW_REPEAT? Seems never happen" << std::endl;
				return false;
			}
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what() << std::endl;
			return false;
		}
	}


	bool Viewer::callback_event_keyboard(int key, int action, int modifiers) {
		try {
			if (action == GLFW_PRESS || action == GLFW_REPEAT) {
				return key_press_event(key, modifiers);
			}
			else {
				return key_release_event(key, modifiers);
			}
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what() << std::endl;
			return false;
		}
	}


	bool Viewer::callback_event_character(unsigned int codepoint) {
		try {
			return char_input_event(codepoint);
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what()
				<< std::endl;
			return false;
		}
	}


	bool Viewer::callback_event_drop(int count, const char **filenames) {
		try {
			std::vector<std::string> arg(count);
			for (int i = 0; i < count; ++i)
				arg[i] = filenames[i];
			return drop_event(arg);
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what()
				<< std::endl;
			return false;
		}
	}


	bool Viewer::callback_event_scroll(double dx, double dy) {
		try {
			return mouse_scroll_event(mouse_x_, mouse_y_, static_cast<int>(dx), static_cast<int>(dy));
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what()
				<< std::endl;
			return false;
		}
	}


	void Viewer::callback_event_resize(int w, int h) {
		if (w == 0 && h == 0)
			return;       

        try {
            width_ = w;
            height_ = h;
            camera_->setScreenWidthAndHeight(w, h);
            glViewport(0, 0, static_cast<int>(w * highdpi_), static_cast<int>(h * highdpi_));
            post_resize(w, h);
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in event handler: " << e.what()
				<< std::endl;
		}
	}


	bool Viewer::focus_event(bool focused) {
		if (focused) {
			// ... 
		}
		return false;
	}


	Viewer::~Viewer() {
		cleanup();
	}


	void Viewer::cleanup() {
		// viewer may have already been destroyed by the user
		if (!window_)
			return;

		if (camera_) {
			delete camera_;
			camera_ = nullptr;
		}

		if (points_program_) {
			delete points_program_;
			points_program_ = nullptr;
		}

		if (lines_program_) {
			delete lines_program_;
			lines_program_ = nullptr;
		}

		if (surface_program_) {
			delete surface_program_;
			surface_program_ = nullptr;
		}

		if (axes_) {
			delete axes_;
			axes_ = nullptr;
		}

		for (auto m : models_)
			delete m;

		glfwDestroyWindow(window_);
		window_ = nullptr;
		glfwTerminate();
	}


	void Viewer::set_title(const std::string &title) {
		if (title != title_) {
			glfwSetWindowTitle(window_, title.c_str());
			title_ = title;
		}
	}


    void Viewer::resize(int w, int h) {
        w = static_cast<int>(w / highdpi_);
        h = static_cast<int>(h / highdpi_);
        glfwSetWindowSize(window_, w, h);
    }


    void Viewer::update() const {
		glfwPostEmptyEvent();
	}


	bool Viewer::mouse_press_event(int x, int y, int button, int modifiers) {
        return false;
	}


	bool Viewer::mouse_release_event(int x, int y, int button, int modifiers) {
		if (button == GLFW_MOUSE_BUTTON_LEFT && modifiers == GLFW_MOD_CONTROL) { // ZOOM_ON_REGION
			int xmin = std::min(mouse_pressed_x_, x);	int xmax = std::max(mouse_pressed_x_, x);
			int ymin = std::min(mouse_pressed_y_, y);	int ymax = std::max(mouse_pressed_y_, y);
			camera_->fitScreenRegion(xmin, ymin, xmax, ymax);
		}

		button_ = -1;
		return false;
	}


	bool Viewer::mouse_drag_event(int x, int y, int dx, int dy, int button, int modifiers) {
		if (modifiers != GLFW_MOD_CONTROL) { // GLFW_MOD_CONTROL is reserved for zoom on region
			switch (button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:
                camera_->frame()->action_rotate(x, y, dx, dy, camera_);
				break;
			case GLFW_MOUSE_BUTTON_RIGHT:
                camera_->frame()->action_translate(x, y, dx, dy, camera_);
				break;
			case GLFW_MOUSE_BUTTON_MIDDLE:
				if (dy != 0)
					camera_->frame()->action_zoom(dy > 0 ? 1 : -1, camera_);
				break;
			}
		}

		return false;
	}


	bool Viewer::mouse_free_move_event(int x, int y, int dx, int dy, int modifiers) {
        // highlight geometry primitives here
		return false;
	}


	bool Viewer::mouse_scroll_event(int x, int y, int dx, int dy) {
		camera_->frame()->action_zoom(dy, camera_);
		return false;
	}


	bool Viewer::key_press_event(int key, int modifiers) {
		if (key == GLFW_KEY_A && modifiers == 0) {
			show_corner_axes_ = !show_corner_axes_;
		}
		else if (key == GLFW_KEY_C && modifiers == 0) {
			if (!models_.empty() && model_idx_ >= 0 && model_idx_ < models_.size()) {
				const Box3& box = models_[model_idx_]->bounding_box();
				camera_->setSceneBoundingBox(box.min(), box.max());
				camera_->showEntireScene();
			}
		}
		else if (key == GLFW_KEY_F && modifiers == 0) {
			if (!models_.empty()) {
				Box3 box;
				for (auto m : models_)
					box.add_box(m->bounding_box());
				camera_->setSceneBoundingBox(box.min(), box.max());
				camera_->showEntireScene();
			}
		}
		else if (key == GLFW_KEY_M && modifiers == 0) {
			// NOTE: switching on/off MSAA in this way will affect all viewers because OpenGL 
			//       is a state machine. For multi-window applications, you have to call 
			//		 glDisable()/glEnable() before the individual draw functions.
			if (samples_ > 0) {
				if (glIsEnabled(GL_MULTISAMPLE)) {
					glDisable(GL_MULTISAMPLE);
					std::cout << title_ + ": MSAA disabled" << std::endl;
				}
				else {
					glEnable(GL_MULTISAMPLE);
					std::cout << title_ + ": MSAA enabled" << std::endl;
				}
			}
		}
		else if (key == GLFW_KEY_F1 && modifiers == 0)
			std::cout << usage() << std::endl;
		else if (key == GLFW_KEY_P && modifiers == 0) {
			if (camera_->type() == Camera::PERSPECTIVE)
				camera_->setType(Camera::ORTHOGRAPHIC);
			else
				camera_->setType(Camera::PERSPECTIVE);
		}
		else if (key == GLFW_KEY_O && modifiers == GLFW_MOD_CONTROL)
			open();
		else if (key == GLFW_KEY_S && modifiers == GLFW_MOD_CONTROL)
			save();

		else if (key == GLFW_KEY_MINUS && modifiers == 0) {
			for (auto m : models_) {
				for (auto d : m->points_drawables()) {
					float size = d->point_size() - 1.0f;
					if (size < 1)
						size = 1;
					d->set_point_size(size);
				}
			}
		}	
		else if (key == GLFW_KEY_EQUAL && modifiers == 0) {
			for (auto m : models_) {
				for (auto d : m->points_drawables()) {
					float size = d->point_size() + 1.0f;
					if (size > 20)
						size = 20;
					d->set_point_size(size);
				}
			}
		}

		else if (key == GLFW_KEY_MINUS && modifiers == GLFW_MOD_CONTROL)
			camera_->frame()->action_zoom(-1, camera_);
		else if (key == GLFW_KEY_EQUAL && modifiers == GLFW_MOD_CONTROL)
			camera_->frame()->action_zoom(1, camera_);

		else if (key == GLFW_KEY_COMMA && modifiers == 0) {
			if (models_.empty())
				model_idx_ = -1;
			else
				model_idx_ = int((model_idx_ - 1 + models_.size()) % models_.size());
			if (model_idx_ >= 0)
				std::cout << "current model: " << model_idx_ << ", " << models_[model_idx_]->name() << std::endl;
		}
		else if (key == GLFW_KEY_PERIOD && modifiers == 0) {
			if (models_.empty())
				model_idx_ = -1;
			else
				model_idx_ = int((model_idx_ + 1) % models_.size());
			if (model_idx_ >= 0)
				std::cout << "current model: " << model_idx_ << ", " << models_[model_idx_]->name() << std::endl;
		}
        else if (key == GLFW_KEY_DELETE && modifiers == 0) {
            if (current_model())
                delete_model(current_model());
        }
		else if (key == GLFW_KEY_W && modifiers == 0) {
			if (model_idx_ < models_.size()) {
				SurfaceMesh* m = dynamic_cast<SurfaceMesh*>(models_[model_idx_]);
				if (m) {
					LinesDrawable* wireframe = m->lines_drawable("wireframe");
					if (!wireframe) {
						wireframe = m->add_lines_drawable("wireframe");
						std::vector<unsigned int> indices;
						for (auto e : m->edges()) {
							SurfaceMesh::Vertex s = m->vertex(e, 0);
							SurfaceMesh::Vertex t = m->vertex(e, 1);
							indices.push_back(s.idx());
							indices.push_back(t.idx());
						}
						auto points = m->get_vertex_property<vec3>("v:point");
						wireframe->update_vertex_buffer(points.vector());
						wireframe->update_index_buffer(indices);
					}
					else
						wireframe->set_visible(!wireframe->is_visible());
				}
			}
		}

		pressed_key_ = key;

		return false;
	}


	bool Viewer::key_release_event(int key, int modifiers) {
		pressed_key_ = GLFW_KEY_UNKNOWN;
		return false;
	}


	bool Viewer::char_input_event(unsigned int codepoint) {
		//switch (codepoint) {
		//case '-':	break;
		//case 'c':	break;
		//case 'C':	break;
		//case '0':	break;
		//default:
		//	return false;
		//}

		return false;
	}


	bool Viewer::drop_event(const std::vector<std::string> & filenames) {
		int count = 0;
		for (auto& name : filenames) {
			if (open(name))
				++count;
		}

		if (count > 0) {
			model_idx_ = static_cast<int>(models_.size()) - 1; // make the last one current
			Box3 box;
			for (auto m : models_)
				box.add_box(m->bounding_box());
			camera_->setSceneBoundingBox(box.min(), box.max());
			camera_->showEntireScene();
			update();
			return true;
		}
		return false;
	}


	vec3 Viewer::point_under_pixel(int x, int y, bool &found) const {

        // GLFW (same as Qt) uses upper corner for its origin while GL uses the lower corner.
        int glx = x;
        int gly = height_ - 1 - y;

        // NOTE: when dealing with OpenGL, we alway work in the highdpi screen space
        glx = static_cast<int>(glx * highdpi_);
        gly = static_cast<int>(gly * highdpi_);

        float depth = std::numeric_limits<float>::max();
        glReadPixels(glx, gly, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);	
		found = depth < 1.0f;
		if (found) {
            vec3 point(float(x), float(y), depth);
            // The input to unprojectedCoordinatesOf() is defined in the screen coordinate system
            point = camera_->unprojectedCoordinatesOf(point);
			return point;
		}
#ifndef NDEBUG
		std::cout << "window size: " << width_ << ", "<< height_ 
			<< "; mouse: " << x << ", " << y << std::endl;
#endif
		return vec3();
	}


	inline double get_seconds() {
		return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
	}


	void Viewer::run() {
        // initialize before showing the window because it can be slow
        init();

        resize(width_, height_);    // Make expected window size
        glfwShowWindow(window_);

		// TODO: make it member variable
		bool is_animating = false;

		try {
			// Rendering loop
			const int num_extra_frames = 5;
			const double animation_max_fps = 30;
			int frame_counter = 0;

			while (!glfwWindowShouldClose(window_)) {
				if (!glfwGetWindowAttrib(window_, GLFW_VISIBLE)) // not visible
					continue;

				double tic = get_seconds();
				pre_draw();
				draw();
				post_draw();
				glfwSwapBuffers(window_);

				if (is_animating || frame_counter++ < num_extra_frames)
				{
					glfwPollEvents();
					// In microseconds
					double duration = 1000000.*(get_seconds() - tic);
					const double min_duration = 1000000. / animation_max_fps;
					if (duration < min_duration)
						std::this_thread::sleep_for(std::chrono::microseconds((int)(min_duration - duration)));
				}
				else {
					/* Wait for mouse/keyboard or empty refresh events */
					glfwWaitEvents();
					frame_counter = 0;
				}
			}

			/* Process events once more */
			glfwPollEvents();
		}
		catch (const std::exception &e) {
			std::cerr << "Caught exception in main loop: " << e.what() << std::endl;
		}

		cleanup();
	}


	void Viewer::init() {
		// create shader programs
		// TODO: have a shader manager to manage all the shaders
		points_program_ = new ShaderProgram("points_color");
		if (points_program_->load_shader_from_code(ShaderProgram::VERTEX, easy3d::shadercode::points_color_vert) &&
			points_program_->load_shader_from_code(ShaderProgram::FRAGMENT, easy3d::shadercode::points_color_frag))
		{
			points_program_->set_attrib_name(ShaderProgram::POSITION, "vtx_position");
			points_program_->set_attrib_name(ShaderProgram::COLOR, "vtx_color");
			points_program_->set_attrib_name(ShaderProgram::NORMAL, "vtx_normal");
			points_program_->link_program();
		}
		else {
			std::cerr << "failed creating shader program for points" << std::endl;
			delete points_program_;
			points_program_ = nullptr;
		}

		lines_program_ = new ShaderProgram("line_color");
		if (lines_program_->load_shader_from_code(ShaderProgram::VERTEX, easy3d::shadercode::lines_color_vert) &&
			lines_program_->load_shader_from_code(ShaderProgram::FRAGMENT, easy3d::shadercode::lines_color_frag))
		{
			lines_program_->set_attrib_name(ShaderProgram::POSITION, "vtx_position");
			lines_program_->set_attrib_name(ShaderProgram::COLOR, "vtx_color");
			lines_program_->link_program();
		}
		else {
			std::cerr << "failed creating shader program for lines" << std::endl;
			delete surface_program_;
			surface_program_ = nullptr;
		}

		surface_program_ = new ShaderProgram("surface_color");
		if (surface_program_->load_shader_from_code(ShaderProgram::VERTEX, easy3d::shadercode::surface_color_vert) &&
			surface_program_->load_shader_from_code(ShaderProgram::FRAGMENT, easy3d::shadercode::surface_color_frag))
		{
			surface_program_->set_attrib_name(ShaderProgram::POSITION, "vtx_position");
			surface_program_->set_attrib_name(ShaderProgram::COLOR, "vtx_color");
			surface_program_->link_program();
		}
		else {
			std::cerr << "failed creating shader program for surfaces" << std::endl;
			delete surface_program_;
			surface_program_ = nullptr;
		}
	}


	std::string Viewer::usage() const {
		return std::string(
			"Easy3D viewer usage:												\n"
			"  F1:              Help											\n"
			"  Ctrl + O:        Open file										\n"
			"  Ctrl + S:        Save file										\n"
            "  Fn + Delete:     Delete current model                            \n"
            "  Left:            Rotate      									\n"
            "  Right:           Translate    									\n"
			"  Middle/Wheel:    Zoom out/in										\n"
            "  Ctrl + '-'/'+':  Zoom out/in										\n"
            "  '+'/'-':         Increase/Decrease point size                    \n"
			"  F:               Fit screen (entire scene/all models)     		\n"
			"  C:               Fit screen (current model only)					\n"
			"  P:               Toggle perspective/orthographic projection)		\n"
			"  A:               Toggle axes										\n"
			"  W:               Toggle wireframe								\n"
			"  < or >:          Switch between models							\n"
			"  M:               Toggle MSAA										\n"
		);
	}


    bool Viewer::open() {
        const std::vector<std::string> filetypes = {"*.ply", "*.obj", "*.off", "*.stl", "*.bin", "*.xyz", "*.bxyz", "*.las", "*.laz", "*.ptx"};
        const std::vector<std::string>& file_names = FileDialog::open(filetypes, true, "");

        int count = 0;
        for (const auto& file_name : file_names) {
            if (open(file_name))
                ++count;
        }
        return count > 0;
    }


    Model* Viewer::open(const std::string& file_name) {
        for (auto m : models_) {
            if (m->name() == file_name) {
                std::cout << "model alreaded loaded: \'" << file_name << std::endl;
                return nullptr;
            }
        }

        std::string ext = file::extension(file_name, true);
        bool is_ply_mesh = false;
        if (ext == "ply")
            is_ply_mesh = (io::PlyReader::num_faces(file_name) > 0);

        Model* model = nullptr;
        if ((ext == "ply" && is_ply_mesh) || ext == "obj" || ext == "off" || ext == "stl" || ext == "plg") { // mesh
            SurfaceMesh* mesh = MeshIO::load(file_name);
            if (mesh) {
                model = mesh;
                create_drawables(mesh);
                std::cout << "mesh loaded. num faces: " << mesh->n_faces() << "; "
                    << "num vertices: " << mesh->n_vertices() << "; "
                    << "num edges: " << mesh->n_edges() << std::endl;
            }
        }
        else if (ext == "mesh" || ext == "meshb" || ext == "tet") { // cgraph
//            model = CGraphIO::read(name);
//            addModel(model, true, fit);
        }
        else { // point cloud
            PointCloud* cloud = PointCloudIO::load(file_name);
            if (cloud) {
                create_drawables(cloud);
                model = cloud;
                std::cout << "cloud loaded. num vertices: " << cloud->n_vertices() << std::endl;
            }
        }

        if (model) {
            model->set_name(file_name);
            add_model(model);
            return model;
        }

        return nullptr;
    }


    void Viewer::create_drawables(Model* m) {
        if (dynamic_cast<PointCloud*>(m)) {
            PointCloud* cloud = dynamic_cast<PointCloud*>(m);
            // create points drawable
            auto points = cloud->get_vertex_property<vec3>("v:point");
            PointsDrawable* drawable = cloud->add_points_drawable("points");
            drawable->update_vertex_buffer(points.vector());
            auto normals = cloud->get_vertex_property<vec3>("v:normal");
            if (normals)
                drawable->update_normal_buffer(normals.vector());
            auto colors = cloud->get_vertex_property<vec3>("v:color");
            if (colors) {
                drawable->update_color_buffer(colors.vector());
                drawable->set_per_vertex_color(true);
            }
        }
        else if (dynamic_cast<SurfaceMesh*>(m)) {
            SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(m);
            FacesDrawable* surface = mesh->add_faces_drawable("surface");
#if 1   // flat shading
            auto points = mesh->get_vertex_property<vec3>("v:point");
            auto colors = mesh->get_vertex_property<vec3>("v:color");

            std::vector<vec3> vertices, vertex_normals, vertex_colors;
            for (auto f : mesh->faces()) {
                // we assume convex polygonal faces and we render in triangles
                SurfaceMesh::Halfedge start = mesh->halfedge(f);
                SurfaceMesh::Halfedge cur = mesh->next_halfedge(mesh->next_halfedge(start));
                SurfaceMesh::Vertex va = mesh->to_vertex(start);
                const vec3& pa = points[va];
                while (cur != start) {
                    SurfaceMesh::Vertex vb = mesh->from_vertex(cur);
                    SurfaceMesh::Vertex vc = mesh->to_vertex(cur);
                    const vec3& pb = points[vb];
                    const vec3& pc = points[vc];
                    vertices.push_back(pa);
                    vertices.push_back(pb);
                    vertices.push_back(pc);

                    const vec3& n = geom::triangle_normal(pa, pb, pc);
                    vertex_normals.insert(vertex_normals.end(), 3, n);

                    if (colors) {
                        vertex_colors.push_back(colors[va]);
                        vertex_colors.push_back(colors[vb]);
                        vertex_colors.push_back(colors[vc]);
                    }
                    cur = mesh->next_halfedge(cur);
                }
            }
            surface->update_vertex_buffer(vertices);
            surface->update_normal_buffer(vertex_normals);
            if (colors)
                surface->update_color_buffer(vertex_colors);
            surface->release_index_buffer();
#else
            auto points = mesh->get_vertex_property<vec3>("v:point");
            surface->update_vertex_buffer(points.vector());
            auto colors = mesh->get_vertex_property<vec3>("v:color");
            if (colors)
                surface->update_color_buffer(colors.vector());

            auto normals = mesh->get_vertex_property<vec3>("v:normal");
            if (normals)
                 surface->update_normal_buffer(normals.vector());
            else {
                std::vector<vec3> normals;
                normals.reserve(mesh->n_vertices());
                for (auto v : mesh->vertices()) {
                    const vec3& n = mesh->compute_vertex_normal(v);
                    normals.push_back(n);
                }
                surface->update_normal_buffer(normals);
            }

            std::vector<unsigned int> indices;
            for (auto f : mesh->faces()) {
                // we assume convex polygonal faces and we render in triangles
                SurfaceMesh::Halfedge start = mesh->halfedge(f);
                SurfaceMesh::Halfedge cur = mesh->next_halfedge(mesh->next_halfedge(start));
                SurfaceMesh::Vertex va = mesh->to_vertex(start);
                while (cur != start) {
                    SurfaceMesh::Vertex vb = mesh->from_vertex(cur);
                    SurfaceMesh::Vertex vc = mesh->to_vertex(cur);
                    indices.push_back(static_cast<unsigned int>(va.idx()));
                    indices.push_back(static_cast<unsigned int>(vb.idx()));
                    indices.push_back(static_cast<unsigned int>(vc.idx()));
                    cur = mesh->next_halfedge(cur);
                }
            }
            surface->update_index_buffer(indices);
#endif
        }
    }

	void Viewer::add_model(Model* model) {
		if (model) {
			unsigned int num = model->n_vertices();
			if (num == 0) {
				std::cerr << "Warning: model does not have vertices. Only complete model can be added to the viewer." << std::endl;
				return;
			}

			if (model->points_drawables().empty() && 
				model->lines_drawables().empty() && 
				model->faces_drawables().empty())
			{
				std::cerr << "Warning: model does not have a drawable (nothing could be rendered). Consider adding drawables before adding it to the viewer." << std::endl;
			}

			Box3 box;
			if (dynamic_cast<PointCloud*>(model)) {
				PointCloud* cloud = dynamic_cast<PointCloud*>(model);
				auto points = cloud->get_vertex_property<vec3>("v:point");
				for (auto v : cloud->vertices())
					box.add_point(points[v]);
				model->set_bounding_box(box);
			}
			else if (dynamic_cast<SurfaceMesh*>(model)) {
				SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(model);
				auto points = mesh->get_vertex_property<vec3>("v:point");
				for (auto v : mesh->vertices())
					box.add_point(points[v]);
			}
			model->set_bounding_box(box);

			models_.push_back(model);
			model_idx_ = static_cast<int>(models_.size()) - 1; // make the last one current

			for (auto m : models_) {
				if (m != model)	// the bbox of model is already contained 
					box.add_box(m->bounding_box());
			}
			camera_->setSceneBoundingBox(box.min(), box.max());
			camera_->showEntireScene();
			update();
		}
	}


    void Viewer::delete_model(Model* model) {
        auto pos = std::find(models_.begin(), models_.end(), model);
        if (pos != models_.end()) {
            models_.erase(pos);
            delete model;
            model_idx_ = static_cast<int>(models_.size()) - 1; // make the last one current
        }
        else
            std::cerr << "no such model: " << model->name() << std::endl;
    }


    Model* Viewer::current_model() const {
        if (models_.empty())
            return nullptr;
        if (model_idx_ < models_.size())
            return models_[model_idx_];
        return nullptr;
    }


	bool Viewer::save() const {
        if (!current_model()) {
            std::cerr << "no model exists" << std::endl;
            return false;
        }

        const std::vector<std::string> filetypes = {"*.ply", "*.obj", "*.off", "*.stl", "*.bin", "*.xyz", "*.bxyz", "*.las", "*.laz"};
        const Model* m = current_model();
        const std::string& file_name = FileDialog::save(filetypes, m->name());
        if (file_name.empty())
            return false;

        bool saved = false;
        if (dynamic_cast<const PointCloud*>(m)) {
            const PointCloud* cloud = dynamic_cast<const PointCloud*>(m);
            saved = PointCloudIO::save(file_name, cloud);
        }
        else if (dynamic_cast<const SurfaceMesh*>(m)) {
            const SurfaceMesh* mesh = dynamic_cast<const SurfaceMesh*>(m);
            saved = MeshIO::save(file_name, mesh);
        }

        if (saved)
            std::cout << "file successfully saved" << std::endl;

        return saved;
	}

	void Viewer::draw_corner_axes() {
		if (!lines_program_)
			return;

		if (!axes_) {
			float len = 0.7f;
			std::vector<vec3> points = { vec3(0,0,0), vec3(len,0,0), vec3(0,0,0), vec3(0,len,0), vec3(0,0,0), vec3(0,0,len) };
			std::vector<vec3> colors = { vec3(1,0,0), vec3(1,0,0), vec3(0,1,0), vec3(0,1,0), vec3(0,0,1), vec3(0,0,1) };
			axes_ = new LinesDrawable("corner_axes");
			axes_->update_vertex_buffer(points);
			axes_->update_color_buffer(colors);
		}
		// The viewport and the scissor are changed to fit the lower left corner.
        int viewport[4], scissor[4];
        glGetIntegerv(GL_VIEWPORT, viewport);	
        glGetIntegerv(GL_SCISSOR_BOX, scissor);	

		static int corner_frame_size = 150;
		glViewport(0, 0, corner_frame_size, corner_frame_size);
		glScissor(0, 0, corner_frame_size, corner_frame_size);	

		// To make the axis appear over other objects: reserve a tiny bit of the 
		// front depth range. NOTE: do remember to restore it later.
        glDepthRange(0, 0.001);

		const mat4& proj = easy3d::ortho(-1, 1, -1, 1, -1, 1);
		const mat4& view = camera_->orientation().inverse().matrix();
		const mat4& MVP = proj * view;
		lines_program_->bind();
		lines_program_->set_uniform("MVP", MVP);
		lines_program_->set_uniform("per_vertex_color", true);
		axes_->draw(false);					
		lines_program_->unbind();

		// restore
		glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        glDepthRange(0.0, 1.0);
	}


	void Viewer::pre_draw() {
		glfwMakeContextCurrent(window_);
		glClearColor(background_color_[0], background_color_[1], background_color_[2], 1.0f);
		glClearDepth(1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}


	void Viewer::post_draw() {
		// Visual hints: axis, camera, grid...
		if (show_corner_axes_)
			draw_corner_axes();
	}


	void Viewer::draw() {
		if (models_.empty())
			return;

		// Let's check if wireframe and surfaces are both shown. If true, we 
		// make the depth coordinates of the surface smaller, so that displaying
		// the mesh and the surface together does not cause Z-fighting.
		std::size_t count = 0;
		for (auto m : models_) {
			if (!m->is_visible())
				continue;
			for (auto d : m->lines_drawables()) {
				if (d->is_visible())
					++count;
			}
		}
		if (count > 0) {
			glEnable(GL_POLYGON_OFFSET_FILL);
			glPolygonOffset(0.5f, -0.0001f);
		}

		surface_program_->bind();	
		const mat4& MVP = camera_->modelViewProjectionMatrix();
		surface_program_->set_uniform("MVP", MVP);		
		// light is defined in VC
		vec4 eyeLightPos(0.27f, 0.27f, 0.92f, 0);
		const mat4& MV = camera_->modelViewMatrix();
		const vec4& wLightPos = inverse(MV) * eyeLightPos;
		surface_program_->set_uniform("wLightPos", wLightPos);		
		// NOTE: camera position is defined in world coordinate system.
		const vec3& wCamPos = camera_->position();
		// it can also be computed as follows:
		//const vec3& wCamPos = invMV * vec4(0, 0, 0, 1);
		surface_program_->set_uniform("wCamPos", wCamPos);		
		for (std::size_t idx = 0; idx < models_.size(); ++idx) {
			Model* m = models_[idx];
			if (!m->is_visible())
				continue;
			for (auto d : m->faces_drawables()) {
				if (d->is_visible()) {
					surface_program_->set_uniform("per_vertex_color", d->per_vertex_color());
					surface_program_->set_uniform("default_color",
					idx == model_idx_ ? d->default_color() : vec3(0.8f, 0.8f, 0.8f));		
					d->draw(false);
				}
			}
		}
		surface_program_->unbind();	

		if (count > 0) 
			glDisable(GL_POLYGON_OFFSET_FILL);

		lines_program_->bind();
		lines_program_->set_uniform("MVP", MVP);
		for (auto m : models_) {
			if (!m->is_visible())
				continue;
			for (auto d : m->lines_drawables()) {
				if (d->is_visible()) {
					lines_program_->set_uniform("per_vertex_color", d->per_vertex_color());
					lines_program_->set_uniform("default_color", d->default_color());
					d->draw(false);
				}
			}
		}
		lines_program_->unbind();

		points_program_->bind();
		points_program_->set_uniform("MVP", MVP);	
		points_program_->set_uniform("wLightPos", wLightPos);	
		points_program_->set_uniform("wCamPos", wCamPos);	
		for (std::size_t idx = 0; idx < models_.size(); ++idx) {
			Model* m = models_[idx];
			if (!m->is_visible())
				continue;
			PointCloud* cloud = dynamic_cast<PointCloud*>(m);
			if (!cloud)
				continue;
			bool lighting = cloud->get_vertex_property<vec3>("v:normal");
			points_program_->set_uniform("lighting", lighting);
			bool per_vertex_color = cloud->get_vertex_property<vec3>("v:color");
			for (auto d : m->points_drawables()) {
				if (d->is_visible()) {
					points_program_->set_uniform("per_vertex_color", d->per_vertex_color());
					points_program_->set_uniform("default_color", d->default_color());		
					d->draw(false);
				}
			}
		}
		points_program_->unbind();	
	}


}
