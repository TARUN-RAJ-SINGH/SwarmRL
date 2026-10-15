/**
 * SwarmRL — Real-Time WebSocket Simulation Service
 * 
 * Manages WebSocket connection, auto-reconnection, event subscriptions,
 * and command dispatching to the FastAPI backend.
 */

export class SimulationSocket {
  constructor(url = 'ws://localhost:8000/ws/simulation') {
    this.url = url;
    this.ws = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1500;
    this.autoReconnect = true;
    this.isConnected = false;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this._emit('connect', { url: this.url });
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const type = payload.type || 'message';
          this._emit(type, payload);
          this._emit('all', payload);
        } catch (err) {
          console.error('[SimulationSocket] Error parsing frame JSON:', err);
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this._emit('disconnect', {});
        if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          setTimeout(() => this.connect(), this.reconnectDelay);
        }
      };

      this.ws.onerror = (err) => {
        this._emit('error', err);
      };
    } catch (err) {
      console.error('[SimulationSocket] Connection error:', err);
    }
  }

  disconnect() {
    this.autoReconnect = false;
    if (this.ws) {
      this.ws.close();
    }
    this.isConnected = false;
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  _emit(event, data) {
    if (this.listeners.has(event)) {
      for (const cb of this.listeners.get(event)) {
        try {
          cb(data);
        } catch (e) {
          console.error(`[SimulationSocket] Listener error for ${event}:`, e);
        }
      }
    }
  }

  send(action, payload = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[SimulationSocket] Cannot send message, socket not open');
      return false;
    }
    const message = JSON.stringify({ action, ...payload });
    this.ws.send(message);
    return true;
  }

  // Helper control methods
  start() {
    return this.send('start');
  }

  pause() {
    return this.send('pause');
  }

  step() {
    return this.send('step');
  }

  reset(seed = null) {
    return this.send('reset', seed !== null ? { seed } : {});
  }

  setPolicy(policy) {
    return this.send('set_policy', { policy });
  }

  setFps(fps) {
    return this.send('set_fps', { fps });
  }

  setNumDrones(count) {
    return this.send('set_num_drones', { count });
  }
}

// Global singleton instance
export const simulationSocket = new SimulationSocket();
export default simulationSocket;
