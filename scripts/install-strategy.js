const { execSync } = require('child_process');

const shouldCompile = process.env.FAST_NORM_COMPILE === 'true';

if (shouldCompile) {
	console.log('[fast-norm] Build flag detected. Compilation from source begins...');
	try {
		execSync('npm run build', { stdio: 'inherit' });
		console.log('[fast-norm] Assembly completed successfully!');
	} catch (error) {
		console.error('[fast-norm] Error when building from source:', error.message);
		process.exit(1);
	}
} else {
	console.log('[fast-norm] Precompiled binaries are used.');
}
