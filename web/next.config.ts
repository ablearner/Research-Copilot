import type { NextConfig } from 'next';
const config:NextConfig={async rewrites(){return [{source:'/api/backend/:path*',destination:`${process.env.PLATFORM_API_ORIGIN || 'http://127.0.0.1:8000'}/:path*`}];}};
export default config;
