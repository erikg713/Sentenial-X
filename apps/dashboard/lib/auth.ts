import axios from "axios";

const AUTH_URL = `${process.env.API_BASE_URL || "http://localhost:8000/api"}/auth/token`;
const TOKEN_KEY = "access_token";

// -------------------------
// Login & Logout
// -------------------------

export async function login(username: string, password: string): Promise<void> {
  try {
    const response = await axios.post(AUTH_URL, new URLSearchParams({
      username,
      password,
    }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    const { access_token } = response.data;
    localStorage.setItem(TOKEN_KEY, access_token);
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || "Login failed");
  }
}

export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
}

// -------------------------
// Token & Auth State
// -------------------------

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

