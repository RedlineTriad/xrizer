use crate::{
    clientcore::{Injected, Injector},
    graphics_backends::{GraphicsBackend, VulkanData, supported_apis_enum},
};
use derive_more::Deref;
use glam::f32::{Quat, Vec3};
use log::{debug, info, warn};
use openvr as vr;
use openxr as xr;
use std::ops::Deref;
use std::sync::{
    RwLock, RwLockReadGuard,
    atomic::{AtomicBool, AtomicI64, Ordering},
};
use std::time::{Duration, Instant};

#[cfg(feature = "monado")]
use openxr_mndx_xdev_space::XR_MNDX_XDEV_SPACE_EXTENSION_NAME;

pub trait Compositor: vr::InterfaceImpl {
    fn post_session_restart(
        &self,
        session: &SessionData,
        waiter: xr::FrameWaiter,
        stream: FrameStream,
    ) -> xr::Result<()>;

    fn get_session_create_info(
        &self,
        data: crate::compositor::CompositorSessionData,
    ) -> SessionCreateInfo;

    #[cfg(test)]
    fn on_restart(&self) {}
}

pub type RealOpenXrData = OpenXrData<crate::compositor::Compositor>;
pub struct OpenXrData<C: Compositor> {
    _entry: xr::Entry,
    pub instance: xr::Instance,
    pub system_id: xr::SystemId,
    pub session_data: SessionLock,
    pub display_time: AtomicXrTime,
    pub display_period_nanos: AtomicI64,
    pub enabled_extensions: xr::ExtensionSet,

    /// should only be externally accessed for testing
    pub(crate) input: Injected<crate::input::Input<C>>,
    pub(crate) compositor: Injected<C>,
}

impl<C: Compositor> Drop for OpenXrData<C> {
    fn drop(&mut self) {
        let session = self
            .session_data
            .data
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(mut data) = session.take() {
            let _ = self.end_session(&mut data);
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Results aren't used, but they're printed
#[allow(clippy::enum_variant_names)]
pub enum InitError {
    EnumeratingExtensionsFailed(xr::sys::Result),
    InstanceCreationFailed(xr::sys::Result),
    SystemCreationFailed(xr::sys::Result),
    SessionCreationFailed(SessionCreationError),
}

impl From<SessionCreationError> for InitError {
    fn from(value: SessionCreationError) -> Self {
        Self::SessionCreationFailed(value)
    }
}

fn get_app_name() -> Option<String> {
    let exe = std::fs::read_link("/proc/self/exe")
        .inspect_err(|e| warn!("Couldn't get app name from /proc/self/exe: {e}"))
        .ok()?;

    let basename = exe.file_name().unwrap();
    if basename == "wine64-preloader" || basename == "wine-preloader" {
        fn extract_wine_exe_name() -> Option<String> {
            let exe_path = std::env::args().next()?;
            // The Windows path separator is \ (instead of /) so we can't use Path.
            // We just want the basename anyway, so we'll just grab the last piece.
            let exe_name = exe_path.rsplit_once('\\')?.1;
            Some(
                exe_name
                    .strip_suffix(".exe")
                    .unwrap_or(exe_name)
                    .to_string(),
            )
        }
        if let Some(name) = extract_wine_exe_name() {
            return Some(name);
        }
    }

    Some(basename.to_string_lossy().into_owned())
}

fn make_version() -> u32 {
    env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap_or(0) * 1000000
        + env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap_or(0) * 1000
        + env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap_or(1)
}

impl<C: Compositor> OpenXrData<C> {
    pub fn new(injector: &Injector) -> Result<Self, InitError> {
        let platform_info = ();

        #[cfg(all(not(test), feature = "static-openxr"))]
        let entry = xr::Entry::linked(&platform_info).expect("Failed to linked OpenXR loader");

        #[cfg(all(not(test), not(feature = "static-openxr")))]
        let entry = unsafe { xr::Entry::load(&platform_info) }
            .expect("Failed to load OpenXR loader — is libopenxr-loader installed?");

        #[cfg(test)]
        let entry = unsafe {
            xr::Entry::from_get_instance_proc_addr(fakexr::get_instance_proc_addr, &platform_info)
        }
        .unwrap();

        let supported_exts = entry
            .enumerate_extensions()
            .map_err(InitError::EnumeratingExtensionsFailed)?;
        let mut exts = xr::ExtensionSet::default();
        exts.khr_vulkan_enable = supported_exts.khr_vulkan_enable;
        exts.khr_opengl_enable = supported_exts.khr_opengl_enable;
        exts.khr_convert_timespec_time = supported_exts.khr_convert_timespec_time;
        exts.ext_hand_tracking = supported_exts.ext_hand_tracking;
        exts.khr_visibility_mask = supported_exts.khr_visibility_mask;
        exts.khr_composition_layer_cylinder = supported_exts.khr_composition_layer_cylinder;
        exts.khr_composition_layer_equirect2 = supported_exts.khr_composition_layer_equirect2;
        exts.khr_composition_layer_color_scale_bias =
            supported_exts.khr_composition_layer_color_scale_bias;
        exts.htc_vive_focus3_controller_interaction =
            supported_exts.htc_vive_focus3_controller_interaction;
        exts.meta_touch_controller_plus = supported_exts.meta_touch_controller_plus;
        exts.fb_display_refresh_rate = supported_exts.fb_display_refresh_rate;
        exts.ext_view_configuration_views_change =
            supported_exts.ext_view_configuration_views_change;

        // Extension that enables simple full body tracking support via generic tracked devices.
        // Available only in the Monado OpenXR runtime.
        #[cfg(feature = "monado")]
        if supported_exts
            .other
            .contains(&[XR_MNDX_XDEV_SPACE_EXTENSION_NAME.as_bytes(), b"\0"].concat())
        {
            exts.other
                .push([XR_MNDX_XDEV_SPACE_EXTENSION_NAME.as_bytes(), b"\0"].concat());
        }

        let instance = entry
            .create_instance(
                &xr::ApplicationInfo {
                    application_name: get_app_name().as_deref().unwrap_or("XRizer"),
                    application_version: 0,
                    engine_name: "XRizer",
                    engine_version: make_version(),
                    ..Default::default()
                },
                &exts,
                &[],
                &platform_info,
            )
            .map_err(InitError::InstanceCreationFailed)?;

        let system_id = instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(InitError::SystemCreationFailed)?;

        let session_data = SessionLock {
            data: RwLock::new(Some(
                SessionData::new(
                    &instance,
                    system_id,
                    vr::ETrackingUniverseOrigin::Standing,
                    None,
                )?
                .0,
            )),
            runtime_lost: AtomicBool::new(false),
        };

        let display_time = if exts.khr_convert_timespec_time {
            instance.now().map(|t| t.as_nanos()).unwrap_or(1i64)
        } else {
            1i64
        };

        Ok(Self {
            _entry: entry,
            instance,
            system_id,
            session_data,
            display_time: AtomicXrTime(display_time.into()), // This will get replaced on the first WaitGetPoses
            display_period_nanos: 11111111.into(), // This will get replaced on the first WaitGetPoses
            enabled_extensions: exts,
            input: injector.inject(),
            compositor: injector.inject(),
        })
    }

    pub(crate) fn is_runtime_lost(&self) -> bool {
        self.session_data.runtime_lost.load(Ordering::Relaxed)
    }

    pub(crate) fn check_xr<T>(&self, operation: &str, result: xr::Result<T>) -> xr::Result<T> {
        result.inspect_err(|error| {
            if matches!(
                *error,
                xr::sys::Result::ERROR_SESSION_LOST | xr::sys::Result::ERROR_INSTANCE_LOST
            ) {
                if !self.session_data.runtime_lost.swap(true, Ordering::Relaxed) {
                    warn!("OpenXR runtime lost during {operation}: {error}");
                }
            } else {
                debug!("OpenXR {operation} failed: {error}");
            }
        })
    }

    pub fn poll_events(&self) -> xr::Result<()> {
        let data = self.session_data.get()?;
        let session = data.session.as_raw();
        if let Some(state) = self.check_xr("event polling", self.poll_events_impl(&data))? {
            drop(data);
            let mut data = self.session_data.data.write().unwrap();
            if let Some(data) = data.as_mut()
                && data.session.as_raw() == session
            {
                data.state = state;
            }
        }
        Ok(())
    }

    fn poll_events_impl(&self, session_data: &SessionData) -> xr::Result<Option<xr::SessionState>> {
        let mut buf = xr::EventDataBuffer::new();
        let mut state = None;
        while let Some(event) = self.instance.poll_event(&mut buf)? {
            match event {
                xr::Event::SessionStateChanged(event) => {
                    if event.session() == session_data.session.as_raw() {
                        state = Some(event.state());
                        info!("OpenXR session state changed: {:?}", event.state());
                        if event.state() == xr::SessionState::LOSS_PENDING {
                            return Err(xr::sys::Result::ERROR_SESSION_LOST);
                        }
                    }
                }
                xr::Event::InstanceLossPending(_) => {
                    return Err(xr::sys::Result::ERROR_INSTANCE_LOST);
                }
                xr::Event::InteractionProfileChanged(event) => {
                    if event.session() == session_data.session.as_raw()
                        && let Some(input) = self.input.get()
                    {
                        input.interaction_profile_changed(session_data)?;
                    }
                }
                _ => {
                    info!("unknown event");
                }
            }
        }

        Ok(state)
    }

    pub fn restart_session(&self) -> Result<(), SessionRestartError> {
        let mut session_guard = self.session_data.data.write().unwrap();
        let old = session_guard
            .as_mut()
            .ok_or(SessionRestartError::NoSession)?;
        self.end_session(old)
            .map_err(SessionRestartError::Shutdown)?;

        let origin = old.current_origin;
        let comp = self
            .compositor
            .get()
            .expect("Session is being restarted, but compositor has not been set up!");

        let info = comp.get_session_create_info(std::mem::take(&mut old.comp_data));

        // Runtimes can require the old session to be destroyed before creating a new one.
        drop(session_guard.take());
        let (session, waiter, stream) =
            match SessionData::new(&self.instance, self.system_id, origin, Some(&info)) {
                Ok(created) => created,
                Err(error) => {
                    let _ = self.check_xr::<()>("session creation", Err(error.xr_result()));
                    return Err(SessionRestartError::Create(error));
                }
            };

        let result = comp.post_session_restart(&session, waiter, stream);
        if !self.is_runtime_lost()
            && let Some(input) = self.input.get()
        {
            input.post_session_restart(&session);
        }
        *session_guard = Some(session);
        result.map_err(SessionRestartError::RestoreFrame)
    }

    pub fn set_tracking_space(&self, space: vr::ETrackingUniverseOrigin) {
        if let Some(data) = self.session_data.data.write().unwrap().as_mut() {
            data.current_origin = space;
        }
    }

    pub fn get_tracking_space(&self) -> vr::ETrackingUniverseOrigin {
        self.session_data
            .get()
            .map_or(vr::ETrackingUniverseOrigin::Standing, |data| {
                data.current_origin
            })
    }

    pub fn reset_tracking_space(&self, origin: vr::ETrackingUniverseOrigin) {
        let mut guard = self.session_data.data.write().unwrap();
        let Some(SessionData {
            session,
            view_space,
            local_space_reference,
            local_space_adjusted,
            stage_space_reference,
            stage_space_adjusted,
            ..
        }) = guard.as_mut().filter(|_| !self.is_runtime_lost())
        else {
            return;
        };

        let reset_space = |ref_space, adjusted_space: &mut xr::Space, ty| {
            let xr::Posef {
                position,
                orientation,
            } = view_space
                .locate(ref_space, self.display_time.get())
                .unwrap()
                .pose;

            // Only set the rotation around the y axis
            let (twist, _) = swing_twist_decomposition(
                Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w),
                Vec3::Y,
            )
            .unwrap_or_else(|| {
                warn!("Couldn't decompose rotation - using identity");
                (Quat::IDENTITY, Quat::IDENTITY)
            });

            *adjusted_space = session
                .create_reference_space(
                    ty,
                    xr::Posef {
                        position,
                        orientation: xr::Quaternionf {
                            x: twist.x,
                            y: twist.y,
                            z: twist.z,
                            w: twist.w,
                        },
                    },
                )
                .unwrap();
        };

        match origin {
            vr::ETrackingUniverseOrigin::RawAndUncalibrated => {
                // RawAndUncalibrated has no calibration to reset
            }
            vr::ETrackingUniverseOrigin::Standing => reset_space(
                stage_space_reference,
                stage_space_adjusted,
                xr::ReferenceSpaceType::STAGE,
            ),
            vr::ETrackingUniverseOrigin::Seated => reset_space(
                local_space_reference,
                local_space_adjusted,
                xr::ReferenceSpaceType::LOCAL,
            ),
        };
    }

    pub fn get_refresh_rate(&self) -> f32 {
        let get_fallback_rate = || {
            Duration::from_nanos(
                self.display_period_nanos
                    .load(Ordering::Relaxed)
                    .try_into()
                    .unwrap(),
            )
            .as_secs_f32()
        };
        if !self.enabled_extensions.fb_display_refresh_rate {
            return get_fallback_rate();
        }

        self.session_data
            .get()
            .and_then(|data| data.session.get_display_refresh_rate())
            .unwrap_or_else(|_| get_fallback_rate())
    }

    fn end_session(&self, session_data: &mut SessionData) -> Result<(), SessionShutdownError> {
        if self.is_runtime_lost() {
            return Err(SessionShutdownError::RuntimeLost);
        }
        if session_data.state == xr::SessionState::EXITING {
            return Err(SessionShutdownError::Exiting);
        }
        if session_data.state != xr::SessionState::STOPPING {
            self.check_xr("request exit", session_data.session.request_exit())
                .map_err(SessionShutdownError::OpenXr)?;
            self.wait_for_session_state(session_data, xr::SessionState::STOPPING)?;
        }
        #[cfg(test)]
        if let Some(comp) = self.compositor.get() {
            comp.on_restart();
        }
        self.check_xr("end session", session_data.session.end())
            .map_err(SessionShutdownError::OpenXr)?;
        self.wait_for_session_state(session_data, xr::SessionState::EXITING)
    }

    fn wait_for_session_state(
        &self,
        session_data: &mut SessionData,
        target: xr::SessionState,
    ) -> Result<(), SessionShutdownError> {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(state) = self
                .check_xr("event polling", self.poll_events_impl(session_data))
                .map_err(SessionShutdownError::OpenXr)?
            {
                session_data.state = state;
            }
            if session_data.state == target {
                return Ok(());
            }
            if session_data.state == xr::SessionState::EXITING {
                return Err(SessionShutdownError::Exiting);
            }
            if Instant::now() >= deadline {
                warn!("Timed out waiting for OpenXR session state {target:?}");
                return Err(SessionShutdownError::Timeout(target));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

pub struct AtomicXrTime(AtomicI64);

impl AtomicXrTime {
    #[inline]
    pub fn set(&self, time: xr::Time) {
        self.0.store(time.as_nanos(), Ordering::Relaxed);
    }

    #[inline]
    pub fn get(&self) -> xr::Time {
        xr::Time::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

pub struct SessionLock {
    data: RwLock<Option<SessionData>>,
    runtime_lost: AtomicBool,
}

impl SessionLock {
    pub fn get(&self) -> xr::Result<SessionReadGuard<'_>> {
        let data = self.data.read().unwrap();
        if self.runtime_lost.load(Ordering::Relaxed) || data.is_none() {
            return Err(xr::sys::Result::ERROR_SESSION_LOST);
        }
        Ok(SessionReadGuard(data))
    }
}

pub struct SessionReadGuard<'a>(RwLockReadGuard<'a, Option<SessionData>>);

impl Deref for SessionReadGuard<'_> {
    type Target = SessionData;

    fn deref(&self) -> &SessionData {
        self.0.as_ref().unwrap()
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Error details are logged with Debug.
pub enum SessionRestartError {
    NoSession,
    /// The previous session is still stored.
    Shutdown(SessionShutdownError),
    /// The previous session was destroyed and no replacement was installed.
    Create(SessionCreationError),
    /// The replacement session was installed before reporting the frame error.
    RestoreFrame(xr::sys::Result),
}

#[derive(Debug)]
#[allow(dead_code)] // Error details are logged with Debug.
pub enum SessionShutdownError {
    RuntimeLost,
    Exiting,
    Timeout(xr::SessionState),
    OpenXr(xr::sys::Result),
}

pub struct Session<G: xr::Graphics> {
    session: xr::Session<G>,
    swapchain_formats: Vec<G::Format>,
}
supported_apis_enum!(pub enum GraphicalSession: Session);
impl std::fmt::Display for GraphicalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphicalSession::Vulkan(_) => f.write_str("GraphicalSession::Vulkan"),
            GraphicalSession::OpenGL(_) => f.write_str("GraphicalSession::OpenGL"),
            #[cfg(test)]
            GraphicalSession::Fake(_) => f.write_str("GraphicalSession::Fake"),
        }
    }
}
supported_apis_enum!(pub enum FrameStream: xr::FrameStream);

// Implementing From results in a "conflicting implementations" error: https://github.com/rust-lang/rust/issues/85576
#[repr(transparent)]
#[derive(Deref)]
pub struct CreateInfo<G: xr::Graphics>(pub G::SessionCreateInfo);
supported_apis_enum!(pub enum SessionCreateInfo: CreateInfo);

impl SessionCreateInfo {
    pub fn from_info<G: xr::Graphics>(info: G::SessionCreateInfo) -> Self
    where
        Self: From<CreateInfo<G>>,
    {
        Self::from(CreateInfo(info))
    }
}

pub struct SessionData {
    pub session: xr::Session<xr::AnyGraphics>,
    session_graphics: GraphicalSession,
    pub state: xr::SessionState,
    pub view_space: xr::Space,
    // The "reference" space is always equivalent to the reference space with an identity offset.
    // The "adjusted" space may have an offset, set by reset_tracking_space.
    // The adjusted spaces should be used for locating things - the reference spaces are only
    // needed for reset_tracking_space
    local_space_reference: xr::Space,
    local_space_adjusted: xr::Space,
    stage_space_reference: xr::Space,
    stage_space_adjusted: xr::Space,
    pub current_origin: vr::ETrackingUniverseOrigin,

    pub input_data: crate::input::InputSessionData,
    pub comp_data: crate::compositor::CompositorSessionData,
    pub overlay_data: crate::overlay::OverlaySessionData,
    /// OpenXR requires graphics information before creating a session, but OpenVR clients don't
    /// have to provide that information until they actually submit a frame. Yet, we need some
    /// information only available behind a session (i.e., calling xrLocateViews for
    /// GetProjectionMatrix), so we will create a session with fake graphics info to appease OpenXR,
    /// that will be replaced with a real one after the application calls IVRSystem::Submit.
    /// When we're using the real session, this will be None.
    /// Note that it also important that this comes after all members which internally use a xr::Session
    /// \- structs are dropped in declaration order, and if we drop our temporary Vulkan data
    /// before the session, the runtime will likely be very unhappy.
    temp_vulkan: Option<VulkanData>,
}

#[derive(Debug)]
#[allow(dead_code)] // Results aren't used, but they're printed
#[allow(clippy::enum_variant_names)]
pub enum SessionCreationError {
    GraphicsSetupFailed(xr::sys::Result),
    SessionCreationFailed(xr::sys::Result),
    ReferenceSpaceCreationFailed(xr::sys::Result),
    PollEventFailed(xr::sys::Result),
    BeginSessionFailed(xr::sys::Result),
}

impl SessionCreationError {
    fn xr_result(&self) -> xr::sys::Result {
        match *self {
            Self::GraphicsSetupFailed(error)
            | Self::SessionCreationFailed(error)
            | Self::ReferenceSpaceCreationFailed(error)
            | Self::PollEventFailed(error)
            | Self::BeginSessionFailed(error) => error,
        }
    }
}

impl SessionData {
    fn new(
        instance: &xr::Instance,
        system_id: xr::SystemId,
        current_origin: vr::ETrackingUniverseOrigin,
        create_info: Option<&SessionCreateInfo>,
    ) -> Result<(Self, xr::FrameWaiter, FrameStream), SessionCreationError> {
        let info;
        let (temp_vulkan, info) = if let Some(info) = create_info {
            if let SessionCreateInfo::Vulkan(info) = info {
                // Monado seems to (incorrectly) give validation errors unless we call this.
                let pd = unsafe { instance.vulkan_graphics_device(system_id, info.instance) }
                    .map_err(SessionCreationError::GraphicsSetupFailed)?;
                assert_eq!(pd, info.physical_device);
            }
            (None, info)
        } else {
            let vk = VulkanData::new_temporary(instance, system_id);
            info = SessionCreateInfo::from_info::<xr::Vulkan>(vk.session_create_info());
            (Some(vk), &info)
        };

        #[macros::any_graphics(SessionCreateInfo)]
        fn create_session<G: xr::Graphics>(
            info: &CreateInfo<G>,
            instance: &xr::Instance,
            system_id: xr::SystemId,
        ) -> xr::Result<(
            xr::Session<xr::AnyGraphics>,
            GraphicalSession,
            xr::FrameWaiter,
            FrameStream,
        )>
        where
            GraphicalSession: From<Session<G>>,
            FrameStream: From<xr::FrameStream<G>>,
        {
            info!(
                "Creating OpenXR session with graphics API {}",
                std::any::type_name::<G>()
            );
            // required to call
            instance.graphics_requirements::<G>(system_id)?;
            let (session, waiter, stream) = unsafe { instance.create_session(system_id, &info.0) }?;
            let swapchain_formats = session.enumerate_swapchain_formats()?;
            Ok((
                session.clone().into_any_graphics(),
                Session {
                    session,
                    swapchain_formats,
                }
                .into(),
                waiter,
                stream.into(),
            ))
        }

        let (session, session_graphics, waiter, stream) = info
            .with_any_graphics::<create_session>((instance, system_id))
            .map_err(SessionCreationError::SessionCreationFailed)?;

        info!("New session created!");
        let create_space = |ty| {
            session
                .create_reference_space(ty, xr::Posef::IDENTITY)
                .map_err(SessionCreationError::ReferenceSpaceCreationFailed)
        };
        let view_space = create_space(xr::ReferenceSpaceType::VIEW)?;
        let local_space_reference = create_space(xr::ReferenceSpaceType::LOCAL)?;
        let local_space_adjusted = create_space(xr::ReferenceSpaceType::LOCAL)?;
        let stage_space_reference = create_space(xr::ReferenceSpaceType::STAGE)?;
        let stage_space_adjusted = create_space(xr::ReferenceSpaceType::STAGE)?;

        let mut buf = xr::EventDataBuffer::new();
        loop {
            match instance
                .poll_event(&mut buf)
                .map_err(SessionCreationError::PollEventFailed)?
            {
                Some(xr::Event::SessionStateChanged(state))
                    if state.session() == session.as_raw() =>
                {
                    match state.state() {
                        xr::SessionState::READY => break,
                        xr::SessionState::LOSS_PENDING => {
                            return Err(SessionCreationError::PollEventFailed(
                                xr::sys::Result::ERROR_SESSION_LOST,
                            ));
                        }
                        xr::SessionState::EXITING => {
                            return Err(SessionCreationError::PollEventFailed(
                                xr::sys::Result::ERROR_SESSION_NOT_RUNNING,
                            ));
                        }
                        _ => {}
                    }
                }
                Some(xr::Event::InstanceLossPending(_)) => {
                    return Err(SessionCreationError::PollEventFailed(
                        xr::sys::Result::ERROR_INSTANCE_LOST,
                    ));
                }
                Some(_) => {}
                None => std::thread::sleep(Duration::from_millis(1)),
            }
        }

        info!(
            "OpenXR session state changed: {:?}",
            xr::SessionState::READY
        );
        session
            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)
            .map_err(SessionCreationError::BeginSessionFailed)?;
        info!("Began OpenXR session.");

        Ok((
            SessionData {
                temp_vulkan,
                session,
                session_graphics,
                state: xr::SessionState::READY,
                view_space,
                local_space_reference,
                local_space_adjusted,
                stage_space_reference,
                stage_space_adjusted,
                input_data: Default::default(),
                comp_data: Default::default(),
                overlay_data: Default::default(),
                current_origin,
            },
            waiter,
            stream,
        ))
    }

    pub fn create_swapchain<G: xr::Graphics>(
        &self,
        info: &xr::SwapchainCreateInfo<G>,
    ) -> xr::Result<xr::Swapchain<G>>
    where
        for<'a> &'a GraphicalSession: TryInto<&'a Session<G>, Error: std::fmt::Display>,
    {
        (&self.session_graphics)
            .try_into()
            .unwrap_or_else(|e| {
                panic!(
                    "Session was not using API {}: {e}",
                    std::any::type_name::<G>()
                )
            })
            .session
            .create_swapchain(info)
    }

    pub fn check_format<G: GraphicsBackend>(&self, info: &mut xr::SwapchainCreateInfo<G::Api>)
    where
        for<'a> &'a GraphicalSession: TryInto<&'a Session<G::Api>, Error: std::fmt::Display>,
        <G::Api as xr::Graphics>::Format: PartialEq,
    {
        let formats = &(&self.session_graphics)
            .try_into()
            .unwrap_or_else(|_| {
                panic!(
                    "Expected session API {}, but current session is using {}!",
                    std::any::type_name::<G>(),
                    self.session_graphics,
                )
            })
            .swapchain_formats;

        if !formats.contains(&info.format) {
            let new_format = formats[0];
            warn!(
                "Requested to init swapchain with unsupported format {:?} - instead using {:?}",
                G::to_nice_format(info.format),
                G::to_nice_format(new_format)
            );
            info.format = new_format;
        }
    }

    pub fn tracking_space(&self) -> &xr::Space {
        self.get_space_for_origin(self.current_origin)
    }

    #[inline]
    pub fn get_space_for_origin(&self, origin: vr::ETrackingUniverseOrigin) -> &xr::Space {
        match origin {
            vr::ETrackingUniverseOrigin::Seated => &self.local_space_adjusted,
            vr::ETrackingUniverseOrigin::Standing => &self.stage_space_adjusted,
            vr::ETrackingUniverseOrigin::RawAndUncalibrated => {
                crate::warn_unimplemented!("RawAndUncalibrated tracking space");
                &self.stage_space_reference
            }
        }
    }

    #[inline]
    pub fn get_space_from_type(&self, ty: xr::ReferenceSpaceType) -> &xr::Space {
        match ty {
            xr::ReferenceSpaceType::VIEW => &self.view_space,
            xr::ReferenceSpaceType::LOCAL => &self.local_space_adjusted,
            xr::ReferenceSpaceType::STAGE => &self.stage_space_adjusted,
            other => panic!("Unsupported reference space type: {other:?}"),
        }
    }

    #[inline]
    pub fn current_origin_as_reference_space(&self) -> xr::ReferenceSpaceType {
        match self.current_origin {
            vr::ETrackingUniverseOrigin::Seated => xr::ReferenceSpaceType::LOCAL,
            vr::ETrackingUniverseOrigin::Standing => xr::ReferenceSpaceType::STAGE,
            vr::ETrackingUniverseOrigin::RawAndUncalibrated => {
                crate::warn_unimplemented!("RawAndUncalibrated tracking space");
                xr::ReferenceSpaceType::STAGE
            }
        }
    }

    /// Returns true if this session is not using a temporary graphics setup.
    #[inline]
    pub fn is_real_session(&self) -> bool {
        self.temp_vulkan.is_none()
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Hand {
    Left = 1,
    Right,
}

impl TryFrom<vr::ETrackedControllerRole> for Hand {
    type Error = ();
    #[inline]
    fn try_from(value: vr::ETrackedControllerRole) -> Result<Self, Self::Error> {
        match value {
            vr::ETrackedControllerRole::LeftHand => Ok(Hand::Left),
            vr::ETrackedControllerRole::RightHand => Ok(Hand::Right),
            _ => Err(()),
        }
    }
}

impl From<Hand> for vr::ETrackedControllerRole {
    fn from(hand: Hand) -> Self {
        match hand {
            Hand::Left => vr::ETrackedControllerRole::LeftHand,
            Hand::Right => vr::ETrackedControllerRole::RightHand,
        }
    }
}

impl From<Hand> for xr::HandEXT {
    fn from(hand: Hand) -> Self {
        match hand {
            Hand::Left => xr::HandEXT::LEFT,
            Hand::Right => xr::HandEXT::RIGHT,
        }
    }
}

/// Taken from: https://github.com/bitshifter/glam-rs/issues/536
/// Decompose the rotation on to 2 parts.
///
/// 1. Twist - rotation around the "direction" vector
/// 2. Swing - rotation around axis that is perpendicular to "direction" vector
///
/// The rotation can be composed back by
/// `rotation = swing * twist`.
/// Order matters!
///
/// has singularity in case of swing_rotation close to 180 degrees rotation.
/// if the input quaternion is of non-unit length, the outputs are non-unit as well
/// otherwise, outputs are both unit
fn swing_twist_decomposition(rotation: Quat, axis: Vec3) -> Option<(Quat, Quat)> {
    let rotation_axis = rotation.xyz();
    let projection = rotation_axis.project_onto(axis);

    let twist = {
        let maybe_flipped_twist = Quat::from_vec4(projection.extend(rotation.w));
        if rotation_axis.dot(projection) < 0.0 {
            -maybe_flipped_twist
        } else {
            maybe_flipped_twist
        }
    };

    if twist.length_squared() != 0.0 {
        let swing = rotation * twist.conjugate();
        Some((twist.normalize(), swing))
    } else {
        None
    }
}

#[cfg(test)]
pub use tests::FakeCompositor;

#[cfg(test)]
mod tests {
    use super::{FrameStream, GraphicsBackend, OpenXrData, SessionCreateInfo};
    use crate::clientcore::Injector;
    use openxr as xr;
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

    pub struct FakeCompositor {
        backend: crate::graphics_backends::VulkanData,
        barrier: Mutex<Option<Arc<Barrier>>>,
        restart_complete: AtomicBool,
        restart_error: Option<xr::sys::Result>,
    }
    impl FakeCompositor {
        pub fn new(xr: &OpenXrData<Self>) -> Self {
            Self::new_with_barrier(xr, None)
        }

        fn new_with_barrier(xr: &OpenXrData<Self>, barrier: Option<Arc<Barrier>>) -> Self {
            Self {
                backend: crate::graphics_backends::VulkanData::new_temporary(
                    &xr.instance,
                    xr.system_id,
                ),
                barrier: barrier.into(),
                restart_complete: false.into(),
                restart_error: None,
            }
        }
    }
    impl openvr::InterfaceImpl for FakeCompositor {
        fn get_version(_: &CStr) -> Option<Box<dyn FnOnce(&Arc<Self>) -> *mut std::ffi::c_void>> {
            None
        }
        fn supported_versions() -> &'static [&'static CStr] {
            &[]
        }
    }
    impl super::Compositor for FakeCompositor {
        fn get_session_create_info(
            &self,
            _: crate::compositor::CompositorSessionData,
        ) -> SessionCreateInfo {
            SessionCreateInfo::from_info::<xr::Vulkan>(self.backend.session_create_info())
        }

        fn post_session_restart(
            &self,
            _: &crate::openxr_data::SessionData,
            _: openxr::FrameWaiter,
            _: FrameStream,
        ) -> xr::Result<()> {
            self.restart_complete.store(true, Ordering::Relaxed);
            self.restart_error.map_or(Ok(()), Err)
        }

        fn on_restart(&self) {
            if let Some(barrier) = self.barrier.lock().unwrap().take() {
                barrier.wait();
            }
        }
    }

    #[test]
    fn shutdown_handles_first_loss_in_request_exit() {
        let data = OpenXrData::<FakeCompositor>::new(&Injector::default()).unwrap();
        let session = data.session_data.get().unwrap().session.as_raw();
        let called = fakexr::fail_next_session_call(
            session,
            fakexr::SessionCall::RequestExit,
            xr::sys::Result::ERROR_INSTANCE_LOST,
        );
        drop(data);
        assert!(called.load(Ordering::Relaxed));
    }

    #[test]
    fn restart_reports_shutdown_error_and_keeps_the_old_session() {
        use super::{SessionRestartError, SessionShutdownError};

        let data = OpenXrData::<FakeCompositor>::new(&Injector::default()).unwrap();
        let session = data.session_data.get().unwrap().session.as_raw();
        fakexr::fail_next_session_call(
            session,
            fakexr::SessionCall::RequestExit,
            xr::sys::Result::ERROR_RUNTIME_FAILURE,
        );

        assert!(matches!(
            data.restart_session(),
            Err(SessionRestartError::Shutdown(SessionShutdownError::OpenXr(
                xr::sys::Result::ERROR_RUNTIME_FAILURE
            )))
        ));
        assert_eq!(data.session_data.get().unwrap().session.as_raw(), session);
    }

    #[test]
    fn frame_restoration_failure_keeps_the_new_session() {
        let data = OpenXrData::<FakeCompositor>::new(&Injector::default()).unwrap();
        let old = data.session_data.get().unwrap().session.as_raw();
        let comp = Arc::new(FakeCompositor {
            restart_error: Some(xr::sys::Result::ERROR_RUNTIME_FAILURE),
            ..FakeCompositor::new(&data)
        });
        data.compositor.set(Arc::downgrade(&comp));

        assert!(matches!(
            data.restart_session(),
            Err(super::SessionRestartError::RestoreFrame(
                xr::sys::Result::ERROR_RUNTIME_FAILURE
            ))
        ));
        assert_ne!(data.session_data.get().unwrap().session.as_raw(), old);
        drop(data); // Session must be dropped before Vulkan data.
    }

    #[test]
    fn session_restart_contended() {
        crate::init_logging();
        let data = Arc::new(OpenXrData::<FakeCompositor>::new(&Injector::default()).unwrap());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let comp = Arc::new(FakeCompositor::new_with_barrier(
            &data,
            Some(barrier.clone()),
        ));
        data.compositor.set(Arc::downgrade(&comp));

        std::thread::scope(|scope| {
            {
                let barrier = barrier.clone();
                let data = data.clone();
                let comp = comp.clone();
                scope.spawn(move || {
                    barrier.wait();
                    let _unused = data.session_data.get().unwrap();
                    log::debug!("acquired");
                    assert!(comp.restart_complete.load(Ordering::Relaxed));
                });
            }

            scope.spawn(|| {
                data.restart_session().unwrap();
            });
        });

        drop(data); // Session must be dropped before Vulkan data.
        drop(comp);
    }
}
