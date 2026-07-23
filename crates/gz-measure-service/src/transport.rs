use crate::wire::measure_fleet_server::MeasureFleetServer;
use crate::{Coordinator, ServiceError, ServiceResult};
use std::future::Future;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};

#[derive(Clone, Debug)]
pub struct CoordinatorTlsConfig {
    pub certificate_pem: Vec<u8>,
    pub private_key_pem: Vec<u8>,
    pub client_ca_pem: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct AgentTlsConfig {
    pub server_ca_pem: Vec<u8>,
    pub certificate_pem: Vec<u8>,
    pub private_key_pem: Vec<u8>,
    pub server_name: String,
}

#[derive(Clone, Copy, Debug)]
pub struct InsecureTransport {
    _private: (),
}

impl InsecureTransport {
    #[must_use]
    pub const fn for_tests() -> Self {
        Self { _private: () }
    }
}

pub struct CoordinatorServer {
    coordinator: Coordinator,
    tls: Option<CoordinatorTlsConfig>,
}

impl CoordinatorServer {
    #[must_use]
    pub fn mutual_tls(coordinator: Coordinator, tls: CoordinatorTlsConfig) -> Self {
        Self {
            coordinator,
            tls: Some(tls),
        }
    }

    #[must_use]
    pub fn insecure_for_tests(coordinator: Coordinator, _insecure: InsecureTransport) -> Self {
        Self {
            coordinator,
            tls: None,
        }
    }

    pub async fn serve(self, listener: TcpListener) -> ServiceResult<()> {
        self.serve_with_shutdown(listener, std::future::pending())
            .await
    }

    pub async fn serve_with_shutdown<F>(
        self,
        listener: TcpListener,
        shutdown: F,
    ) -> ServiceResult<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let reaper = self.coordinator.clone();
        let reaper_interval = reaper_interval();
        let reaper_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(reaper_interval);
            loop {
                interval.tick().await;
                reaper.reap_expired();
            }
        });

        let mut server = Server::builder();
        if let Some(tls) = self.tls {
            let identity = Identity::from_pem(tls.certificate_pem, tls.private_key_pem);
            let client_ca = Certificate::from_pem(tls.client_ca_pem);
            server = server
                .tls_config(
                    ServerTlsConfig::new()
                        .identity(identity)
                        .client_ca_root(client_ca),
                )
                .map_err(|error| ServiceError::configuration(error.to_string()))?;
        }

        let result = server
            .add_service(MeasureFleetServer::new(self.coordinator))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
            .await
            .map_err(|error| ServiceError::transport(error.to_string()));
        reaper_task.abort();
        result
    }
}

fn reaper_interval() -> Duration {
    Duration::from_millis(250)
}
