# LearnableKGE - Loss Landscape Visualization

A Next.js web application for visualizing loss landscapes in 2D and 3D, with support for training trajectory visualization.

## Features

- 📊 2D and 3D loss landscape visualization
- 🎯 Interactive 3D rendering with Three.js
- 📈 Training trajectory overlay
- 💾 DuckDB for efficient data storage
- ⚙️ Support for config files and run directories

## Getting Started

### Prerequisites

- Node.js 18+ 
- Python 3.11+ (for backend loss landscape generation)
- PINN_BDKE project (parent directory)

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. **From Config File**: Upload a `config.yml` file to generate loss landscape
2. **From Run Directory**: Select a training run directory to visualize the training trajectory

## Architecture

- **Frontend**: Next.js + React + Three.js
- **Backend API**: Next.js API routes calling Python backend
- **Database**: DuckDB for storing loss landscape data
- **3D Rendering**: react-three-fiber + drei

