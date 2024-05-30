"use client";
import { Bar, BarChart, CartesianGrid, Legend, RadialBar, RadialBarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function MetricsPage() {
    const responseTimeData = [
        { name: 'Test 1', responseTime: 200 },
        { name: 'Test 2', responseTime: 180 },
        { name: 'Test 3', responseTime: 220 },
        { name: 'Test 4', responseTime: 160 },
        { name: 'Test 5', responseTime: 210 },
    ];
    const currentOutputData = [
        { name: 'PSNR', value: 32, fill: '#8884d8' },
        { name: 'Response Time', value: 180, fill: '#82ca9d' },
      ];
    return (
        <section className="flex min-h-screen flex-col items-center justify-between p-2">
            <h1 className="text-3xl font-bold text-center">Metrics</h1>
            <h2>Response Time</h2>
            <ResponsiveContainer width={500} height={300}>
                <BarChart
                data={responseTimeData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="responseTime" fill="#82ca9d" />
                </BarChart>
            </ResponsiveContainer>
        </section>
    );
}
