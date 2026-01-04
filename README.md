# Physical AI & Humanoid Robotics Textbook

A Docusaurus-based intelligent textbook covering Physical AI and Humanoid Robotics, featuring an integrated RAG (Retrieval-Augmented Generation) chatbot.

## Overview

This repository contains the frontend and content for the Physical AI & Humanoid Robotics textbook. The chatbot functionality is powered by a separately deployed backend.

## Features

- **Intelligent Textbook**: Comprehensive content on Physical AI and Robotics.
- **RAG Chatbot**: An integrated assistant that helps students with textbook content.
- **Modern UI**: Built with Docusaurus for a seamless reading experience.

## Getting Started

### Prerequisites

- Node.js (Version 18.0 or higher)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

Your site will be available at `http://localhost:3000`.

## Configuration

The chatbot requires a backend API URL. This can be configured using environment variables:

- `REACT_APP_API_URL`: The URL of the deployed RAG backend.

In production (e.g., Vercel), set this environment variable in your project settings.

## Deployment

The site is optimized for deployment on [Vercel](https://vercel.com).

1. Connect your repository to Vercel.
2. Add the `REACT_APP_API_URL` environment variable.
3. Vercel will automatically build and deploy your site.

## Documentation

For more information on Docusaurus, visit [docusaurus.io](https://docusaurus.io/).