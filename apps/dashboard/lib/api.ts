import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";

const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000/api";

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Optional: Inject auth token into headers if available
apiClient.interceptors.request.use(
  (config: AxiosRequestConfig) => {
    const token = localStorage.getItem("access_token"); // Or other storage
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Generic GET helper
export async function get<T>(url: string, params?: object): Promise<T> {
  const response: AxiosResponse<T> = await apiClient.get(url, { params });
  return response.data;
}

// Generic POST helper
export async function post<T>(url: string, data?: object): Promise<T> {
  const response: AxiosResponse<T> = await apiClient.post(url, data);
  return response.data;
}

// Example: Fetch threats
export async function fetchThreats() {
  return get<Threat[]>("/threats");
}

// Define Threat type if you want here or import from types
export interface Threat {
  id: string;
  title: string;
  description: string;
  timestamp: string;
  severity: "Low" | "Medium" | "High" | "Critical";
}

