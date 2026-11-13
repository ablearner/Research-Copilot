import {NextResponse} from 'next/server';
// Private files are only served by the authorized backend endpoint.
export async function GET(){return new NextResponse('Not found',{status:404});}
