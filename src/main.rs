#![warn(missing_docs)]
//! Preprocess images from acquire.py, and feed them to cam.py.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use clap::{ArgAction, Parser};
use colored::Colorize;
use figment::{
    providers::{Format, Serialized, Toml},
    Figment,
};
use flexi_logger::{LogSpecification, Logger};
use log::{debug, info, warn};
use ndarray::{s, Array2};
use notify::{RecursiveMode, Watcher};
use notify_debouncer_full::{self, DebouncedEvent};
use serde::{Deserialize, Serialize};
use std::fs;
use std::option::Option;
use std::sync::mpsc;

#[derive(Debug, Parser, Serialize)]
struct Cli {
    /// Input path
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    inpath: Option<String>,

    /// Output path
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    outpath: Option<String>,

    /// Verbosity (-v for info level, -vv for debug)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Quiet output (overrides -v)
    #[arg(long, short)]
    quiet: bool,

    /// Processor name
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    proc: Option<String>,
}

/// Holder for configuration
#[derive(Serialize, Deserialize)]
struct Config {
    /// Input folder path
    inpath: String,
    /// Output folder path
    outpath: String,
    /// Verbosity
    verbose: u8,
    /// Quiet (overrides verbose)
    quiet: bool,
    /// Processor name
    proc: String,
}

#[derive(Debug)]
struct SisImg {
    height: usize,
    width: usize,
    image: Vec<u16>,
}

impl SisImg {
    fn new(arr: Array2<u16>) -> Result<SisImg> {
        let shape = arr.shape();
        let height = shape[0];
        let width = shape[1];

        if height > u16::MAX as usize {
            bail!("Height of image too big ({} > {})", height, u16::MAX);
        }

        if width > u16::MAX as usize {
            bail!("Width of image too big ({} > {})", width, u16::MAX);
        }

        let image = arr.into_raw_vec();
        Ok(SisImg {
            height,
            width,
            image,
        })
    }

    fn read(path: &PathBuf) -> Result<SisImg> {
        debug!("Reading sis image from {:?}", path);
        let mut file = File::open(path)?;

        // First ten bytes are empty
        file.seek(SeekFrom::Start(10))?;

        // Height is a 16 bit integer
        let mut heightbuf = [0u8; 2];
        file.read_exact(&mut heightbuf)?;
        let height = usize::from(u16::from_le_bytes(heightbuf));
        debug!("Image height: {}", height);

        // Width is another 64 bit integer
        let mut widthbuf = [0u8; 2];
        file.read_exact(&mut widthbuf)?;
        let width = usize::from(u16::from_le_bytes(widthbuf));
        debug!("Image width: {}", height);

        // Then there are 186 empty bytes
        file.seek(SeekFrom::Current(186))?;

        let len = height * width;
        let mut image: Vec<u16> = vec![0; len];
        file.read_u16_into::<LittleEndian>(&mut image)?;

        Ok(SisImg {
            height,
            width,
            image,
        })
    }

    fn write(&self, path: PathBuf) -> Result<()> {
        debug!("Writing sis image to path {:?}", path);
        let mut file = File::create(path)?;
        for _ in 0..10 {
            file.write(b" ")?;
        }

        let height = self.height as u16;
        let width = self.width as u16;

        file.write(&height.to_le_bytes())?;
        file.write(&width.to_le_bytes())?;

        for _ in 0..186 {
            file.write(b" ")?;
        }

        let nbytes = 2 * self.height as u32 * self.width as u32;
        let mut imgbuf: Vec<u8> = vec![0; nbytes as usize];
        LittleEndian::write_u16_into(&self.image, &mut imgbuf);

        file.write_all(&imgbuf)?;

        Ok(())
    }
}

impl From<SisImg> for Array2<u16> {
    fn from(value: SisImg) -> Self {
        Array2::from_shape_vec((value.height, value.width), value.image)
            .unwrap()
    }
}

/// Common trait for processors.
///
/// Each processor is just a thin layer over the proc function, which implements
/// all of the logic
trait Process {
    /// Process the files in paths according to processor logic.
    fn proc(&self, paths: Vec<PathBuf>) -> Result<()>;
}

/// This process just copies the files from input to output.
#[derive(Debug, Clone)]
struct Identity {
    outpath: String,
}

impl Identity {
    /// Create a new identity processor with specified paths for input and
    /// output.
    fn new(outpath: &str) -> Identity {
        debug!("Identity processor created with outpath {}", outpath);
        Identity {
            outpath: String::from(outpath),
        }
    }

    fn filecp(&self, path: PathBuf) -> Result<()> {
        debug!("Identity processor function.\n\tPath: {:?}", path);
        let fname = path.file_name();
        if fname.is_none() {
            bail!("Path {:?} is file, but cannot extract filename.", path);
        }
        let fname = fname.unwrap();

        let mut outname = PathBuf::from(self.outpath.clone());
        outname.push(fname);
        debug!("Output filename: {:?}", outname);

        let errstr = format!(
            "Error while copying {:?} to {:?} in Identity type processing",
            path, outname,
        );
        let infostr = format!("Copied {:?} to {:?}", path, outname);

        fs::copy(path, outname).context(errstr)?;
        info!("{}", infostr);

        Ok(())
    }
}

impl Process for Identity {
    fn proc(&self, paths: Vec<PathBuf>) -> Result<()> {
        for p in paths {
            self.filecp(p)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FKSpecies {
    outpath: String,
}

impl FKSpecies {
    fn new(outpath: &str) -> FKSpecies {
        debug!("FKSpecies processor created with outpath {}", outpath);
        FKSpecies {
            outpath: String::from(outpath),
        }
    }

    fn findpattern(paths: Vec<PathBuf>, pattern: &str) -> Result<PathBuf> {
        debug!("Finding pattern {} in {:?}", pattern, paths);
        let imgp = paths
            .iter()
            .filter(|x| x.to_string_lossy().contains(pattern))
            .collect::<Vec<&PathBuf>>();

        if imgp.len() == 0 {
            bail!("Cannot find pattern {} in {:?}", pattern, paths)
        } else {
            let p = imgp[0];
            Ok(p.clone())
        }
    }

    fn calc_od(
        img1: &Array2<u16>,
        img2: &Array2<u16>,
        img3: &Array2<u16>,
    ) -> Array2<f32> {
        // subtract offset
        debug!("Calculating OD from images.");
        let mut img1s: Array2<f32> = (img1 - img3).mapv(|x| f32::from(x));
        let mut img2s: Array2<f32> = (img2 - img3).mapv(|x| f32::from(x));
        let mut output = Array2::<f32>::zeros(img1.raw_dim());

        let height = img1s.shape()[0];
        debug!("Image height {} px", height);

        img1s.par_mapv_inplace(f32::ln);
        let img1s_at = &img1s.slice(s![..height / 2, ..]);
        let img1s_br = &img1s.slice(s![height / 2.., ..]);
        output
            .slice_mut(s![..height / 2, ..])
            .assign(&(img1s_br - img1s_at));

        img2s.par_mapv_inplace(f32::ln);
        let img2s_at = &img2s.slice(s![..height / 2, ..]);
        let img2s_br = &img2s.slice(s![height / 2.., ..]);
        output
            .slice_mut(s![height / 2.., ..])
            .assign(&(img2s_br - img2s_at));

        output
    }
}

impl Process for FKSpecies {
    fn proc(&self, paths: Vec<PathBuf>) -> Result<()> {
        let img1p = FKSpecies::findpattern(paths.clone(), "rawimg-0001")?;
        let img1fn = img1p
            .file_name()
            .ok_or(anyhow!("Cannot find file name in path {:?}", img1p))?;
        debug!("Filename of image 1: {:?}", img1fn);
        let img1op = PathBuf::from(&self.outpath).with_file_name(img1fn);
        debug!("Image 1 will output to: {:?}", img1op);

        let img2p = FKSpecies::findpattern(paths.clone(), "rawimg-0002")?;
        let img2fn = img2p
            .file_name()
            .ok_or(anyhow!("Cannot find file name in path {:?}", img2p))?;
        debug!("Filename of image 2: {:?}", img2fn);
        let img2op = PathBuf::from(&self.outpath).with_file_name(img2fn);
        debug!("Image 2 will output to: {:?}", img2op);

        let img3p = FKSpecies::findpattern(paths.clone(), "rawimg-0003")?;
        let img3fn = img3p
            .file_name()
            .ok_or(anyhow!("Cannot find file name in path {:?}", img3p))?;
        debug!("Filename of image 3: {:?}", img3fn);
        let img3op = PathBuf::from(&self.outpath).with_file_name(img3fn);
        debug!("Image 3 will output to: {:?}", img3op);

        let img1: Array2<u16> = SisImg::read(&img1p)?.into();
        let img2: Array2<u16> = SisImg::read(&img2p)?.into();
        let img3: Array2<u16> = SisImg::read(&img3p)?.into();

        let imgod = (FKSpecies::calc_od(&img1, &img2, &img3) + 1.0) * 1000.0;
        let imgod: Array2<u16> = imgod.mapv(|x| x as u16);

        debug!("Copying raw images to their respective output paths");
        fs::copy(img1p, img1op)?;
        fs::copy(img2p, img2op)?;
        fs::copy(img3p, img3op)?;

        let imgodop = PathBuf::from(&self.outpath)
            .with_file_name("20140000-img-0000.sis");

        debug!("Writing OD image to its path");
        SisImg::new(imgod)?.write(imgodop)?;

        Ok(())
    }
}

/// Call the process function on the debounced event, once for every distinct
/// file path
fn handle_events(
    proc: &Box<dyn Process>,
    events: Vec<DebouncedEvent>,
) -> Result<()> {
    let mut paths = vec![];
    for ev in events {
        for p in ev.paths.clone() {
            paths.push(p);
        }
    }
    paths.dedup();
    debug!("Event paths: {:?}", paths);
    proc.proc(paths)
}

/// Get properly overridden logging level.
///
/// Logging level behaviour from Cli config is:
/// none for warn level
/// -v for info level
/// -vv or more for debug level
/// -q for turning off (overrides any -v)
fn getloglvl(conf: &Config) -> LogSpecification {
    if conf.quiet {
        LogSpecification::off()
    } else {
        match conf.verbose {
            0 => LogSpecification::warn(),
            1 => LogSpecification::info(),
            _ => LogSpecification::debug(),
        }
    }
}

/// Check that specified filepaths are not identical, and that they are folders.
fn checkpaths(conf: &Config) -> Result<()> {
    debug!("Checking paths.");

    if conf.inpath == conf.outpath {
        bail!("Input path and output path must not be identical.");
    }

    if !Path::new(&conf.inpath).is_dir() {
        bail!("Input path must be a directory.");
    }

    if !Path::new(&conf.outpath).is_dir() {
        bail!("Output path must be a directory.");
    }

    Ok(())
}

/// Get the processor selected by the user
fn getproc(conf: &Config) -> Result<Box<dyn Process>> {
    // I swear I tried to make this better, but I couldn't.
    let procs = vec![String::from("identity"), String::from("dummy")];
    if conf.proc == "identity" {
        Ok(Box::new(Identity::new(&conf.outpath)))
    } else if conf.proc == "fkspecies" {
        Ok(Box::new(FKSpecies::new(&conf.outpath)))
    } else {
        bail!(
            "Processor {} unknown, possible values are {:?}",
            conf.proc,
            procs
        )
    }
}

fn main() -> Result<()> {
    let conf: Config = Figment::new()
        .merge(Toml::file("conf/default.toml"))
        .merge(Serialized::defaults(Cli::parse()))
        .extract()?;

    let loglvl = getloglvl(&conf);
    let _logger = Logger::with(loglvl)
        .start()
        .unwrap_or_else(|e| panic!("Cannot start logger. Error:\n{}", e));

    checkpaths(&conf)?;

    let processor = getproc(&conf)?;
    warn!("Chosen processor: {}", &conf.proc);

    let inpath = Path::new(&conf.inpath);

    let (tx, rx) = mpsc::channel();

    let mut debouncer = notify_debouncer_full::new_debouncer(
        Duration::from_millis(200),
        None,
        tx,
    )?;

    let watcher = debouncer.watcher();

    watcher.watch(inpath, RecursiveMode::Recursive)?;

    if !conf.quiet {
        println!("{} {}", "Watching path: ".blue(), conf.inpath.blue());
    }

    for res in rx {
        match res {
            Ok(events) => {
                handle_events(&processor, events)?;
            }
            Err(e) => bail!("Error while processing events:\n\t{:?}", e),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Array2, PathBuf, SisImg};

    #[test]
    fn test_write_read_sis() {
        let path = PathBuf::from("./test/write_sis.sis");
        let imgbuf = Array2::<u16>::eye(4);
        SisImg::new(imgbuf.clone())
            .unwrap()
            .write(path.clone())
            .unwrap();

        let img = SisImg::read(&path).unwrap();
        assert!(img.image == imgbuf.into_raw_vec());
    }
}
