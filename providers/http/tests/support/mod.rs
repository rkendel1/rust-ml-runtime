//! Deterministic in-process HTTP/1.1 mock for OpenAI-compatible tests.
//!
//! No external network or credentials are involved. Responses are written in
//! caller-chosen byte chunks so SSE framing can be split at arbitrary points.
#![allow(dead_code)]

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

#[derive(Clone, Debug)]
pub struct Captured {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Captured {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

#[derive(Clone)]
pub enum Reply {
    /// Complete response with a content-length.
    Json(u16, String),
    /// Streamed response: each chunk is written, then the pause elapses.
    /// With `hold_open` the connection stays open until the client leaves.
    Sse {
        chunks: Vec<Vec<u8>>,
        pause: Duration,
        hold_open: bool,
    },
    /// Accept the request and never answer.
    Stall,
}

pub struct Mock {
    pub url: String,
    pub requests: Arc<Mutex<Vec<Captured>>>,
    /// Set when a client connection that was being streamed to went away.
    pub client_disconnected: Arc<AtomicBool>,
    pub connections: Arc<AtomicUsize>,
}

impl Mock {
    pub async fn start(handler: impl Fn(&Captured) -> Reply + Send + Sync + 'static) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let disconnected = Arc::new(AtomicBool::new(false));
        let connections = Arc::new(AtomicUsize::new(0));
        let handler = Arc::new(handler);
        let (r, d, c) = (requests.clone(), disconnected.clone(), connections.clone());
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                c.fetch_add(1, Ordering::SeqCst);
                let (handler, r, d) = (handler.clone(), r.clone(), d.clone());
                tokio::spawn(async move {
                    let _ = serve(socket, handler, r, d).await;
                });
            }
        });
        Mock {
            url,
            requests,
            client_disconnected: disconnected,
            connections,
        }
    }

    pub fn captured(&self) -> Vec<Captured> {
        self.requests.lock().unwrap().clone()
    }

    pub async fn wait_disconnected(&self) -> bool {
        for _ in 0..200 {
            if self.client_disconnected.load(Ordering::SeqCst) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }
}

async fn read_request(socket: &mut TcpStream) -> std::io::Result<Captured> {
    let mut buffer = Vec::new();
    let head_end = loop {
        let mut chunk = [0u8; 4096];
        let n = socket.read(&mut chunk).await?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(i) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.lines();
    let mut request_line = lines.next().unwrap_or_default().split_whitespace();
    let method = request_line.next().unwrap_or_default().to_owned();
    let path = request_line.next().unwrap_or_default().to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buffer[head_end..].to_vec();
    while body.len() < length {
        let mut chunk = [0u8; 4096];
        let n = socket.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Ok(Captured {
        method,
        path,
        headers,
        body,
    })
}

async fn serve(
    mut socket: TcpStream,
    handler: Arc<dyn Fn(&Captured) -> Reply + Send + Sync>,
    requests: Arc<Mutex<Vec<Captured>>>,
    disconnected: Arc<AtomicBool>,
) -> std::io::Result<()> {
    let request = read_request(&mut socket).await?;
    let reply = handler(&request);
    requests.lock().unwrap().push(request);
    match reply {
        Reply::Json(status, body) => {
            let head = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await?;
            socket.write_all(body.as_bytes()).await?;
            socket.flush().await
        }
        Reply::Sse {
            chunks,
            pause,
            hold_open,
        } => {
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
                .await?;
            for chunk in chunks {
                if socket.write_all(&chunk).await.is_err() || socket.flush().await.is_err() {
                    disconnected.store(true, Ordering::SeqCst);
                    return Ok(());
                }
                tokio::time::sleep(pause).await;
            }
            if hold_open {
                let mut sink = [0u8; 64];
                loop {
                    match socket.read(&mut sink).await {
                        Ok(0) | Err(_) => {
                            disconnected.store(true, Ordering::SeqCst);
                            return Ok(());
                        }
                        Ok(_) => {}
                    }
                }
            }
            Ok(())
        }
        Reply::Stall => {
            let mut sink = [0u8; 64];
            while matches!(socket.read(&mut sink).await, Ok(n) if n > 0) {}
            disconnected.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
}
