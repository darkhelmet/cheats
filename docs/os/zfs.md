# ZFS

Quick reference for OpenZFS pool and filesystem administration. Examples use a pool named `tank`, a dataset named `tank/data`, and Linux persistent disk paths. Replace these with your own names and devices. Run administrative commands as root or with `sudo`; installation and device naming vary by operating system.

## Core Concepts

| Term | Meaning |
| --- | --- |
| Pool (`zpool`) | Storage assembled from one or more virtual devices |
| Vdev | A disk, mirror, or RAIDZ group; losing any top-level data vdev loses the pool |
| Dataset (`zfs`) | A filesystem, volume, snapshot, or other ZFS-managed object |
| Zvol | A dataset exposed as a block device |
| Snapshot | Read-only point-in-time view that retains blocks as live data changes |
| Clone | Writable dataset initially sharing blocks with a snapshot |
| Scrub | Reads allocated data, verifies checksums, and repairs from redundant copies when possible |
| Resilver | Rebuilds missing or outdated data on a replacement or reattached device |

Mirrors survive all but one disk failing **within each mirror**. RAIDZ1, RAIDZ2, and RAIDZ3 tolerate one, two, and three disk failures respectively within each RAIDZ vdev. Redundancy and snapshots do not replace an independent backup.

## Inspect Pools and Datasets

```bash
zpool list                              # Pool capacity and health
zpool status -v                         # Device state and data errors
zpool status -P tank                    # Full device paths
zpool iostat -v tank 2                  # Per-vdev I/O every two seconds
zpool history tank                     # Administrative command history
zpool get all tank                     # Pool properties

zfs list                               # Filesystem and volume usage
zfs list -r tank                        # Include child datasets
zfs list -t snapshot -r tank            # Snapshots
zfs list -o name,used,avail,refer,mountpoint
zfs get compression,compressratio,mountpoint tank/data
zfs get -r -o name,property,value,source compression tank
```

`USED` includes space charged to descendants and snapshots; `REFER` is the data referenced by that dataset. Deleting a file may not free pool space while snapshots still reference its blocks. Pool capacity and dataset available space differ because of redundancy, reservations, quotas, and internal overhead.

## Create a Pool

**Pool creation overwrites the selected devices.** Verify disk identities first. The following layouts are alternatives, not a sequence. Use persistent device identifiers rather than `/dev/sdX` names on Linux.

```bash
# Preview a two-disk mirror without creating it
zpool create -n -o ashift=12 tank mirror \
  /dev/disk/by-id/DISK_A /dev/disk/by-id/DISK_B

# Create the mirror; enable compression for its datasets
zpool create -o ashift=12 -O compression=lz4 -m /tank tank mirror \
  /dev/disk/by-id/DISK_A /dev/disk/by-id/DISK_B

# Alternative: four-disk RAIDZ2 (two parity disks)
zpool create -o ashift=12 -O compression=lz4 -m /tank tank raidz2 \
  /dev/disk/by-id/DISK_A /dev/disk/by-id/DISK_B \
  /dev/disk/by-id/DISK_C /dev/disk/by-id/DISK_D
```

`ashift=12` uses 4 KiB sectors; choose an appropriate value for the devices before creating the vdev. Avoid `-f` unless you understand why ZFS rejected the devices.

## Datasets and Properties

```bash
zfs create tank/data
zfs create -p tank/projects/app         # Create missing parent datasets
zfs create -o mountpoint=/srv/backups tank/backups
zfs rename tank/projects/app tank/projects/service

zfs set compression=lz4 tank/data       # Applies to newly written blocks
zfs set atime=off tank/data             # Disable access-time updates
zfs set recordsize=1M tank/backups      # Large sequential files; new files
zfs set quota=500G tank/data            # Limit dataset + descendants + snapshots
zfs set refquota=400G tank/data         # Limit live dataset, excluding snapshots/children
zfs set reservation=100G tank/data      # Guarantee space for this dataset tree
zfs set quota=none tank/data            # Remove quota
zfs inherit compression tank/data      # Return to inherited/default property

zfs mount tank/data
zfs unmount tank/data
zfs mount -a                           # Mount eligible ZFS filesystems

zfs create -V 20G tank/vm-disk          # Create a block volume (zvol)
```

Properties normally inherit from parent datasets unless overridden. Tune `recordsize` to the workload; it is a maximum filesystem block size, not a disk sector size. Keep free pool capacity for healthy write performance. Avoid enabling deduplication without measuring its memory and storage costs.

## Snapshots and Recovery

```bash
zfs snapshot tank/data@before-update
zfs snapshot -r tank@daily              # Snapshot pool root and all descendants
zfs list -t snapshot -o name,creation,used,refer -s creation
zfs diff tank/data@before-update tank/data

# Read or copy individual files from a mounted filesystem snapshot
ls /tank/data/.zfs/snapshot/before-update/
cp /tank/data/.zfs/snapshot/before-update/report.txt /tmp/report-restored.txt

# Create a writable copy without changing the original dataset
zfs clone tank/data@before-update tank/data-recovery

# Prevent a snapshot from being deleted until its hold is released
zfs hold keep tank/data@before-update
zfs holds tank/data@before-update
zfs release keep tank/data@before-update
```

The `.zfs` directory is hidden from directory listings by default but can be accessed directly. Use the dataset's actual mountpoint. Coordinate snapshots with databases or applications when application-consistent backups are required.

```bash
# DESTRUCTIVE: discard changes since the latest snapshot
zfs rollback tank/data@before-update

# Preview deletion, then delete a snapshot when it is no longer needed
zfs destroy -nv tank/data@before-update
zfs destroy tank/data@before-update
```

Rollback requires the most recent snapshot unless newer snapshots are removed. `zfs rollback -r` destroys newer snapshots and bookmarks; avoid it unless that loss is intended. A clone depends on its origin snapshot, which cannot normally be deleted while the clone exists.

## Send and Receive (Backups)

These examples assume an existing second pool named `backup`. The initial destination dataset `backup/data` must not already exist. `-u` receives without mounting.

```bash
# Initial full backup
zfs snapshot tank/data@backup-1
zfs send tank/data@backup-1 | zfs receive -u backup/data

# Incremental backup: destination must retain the matching base snapshot
zfs snapshot tank/data@backup-2
zfs send -i tank/data@backup-1 tank/data@backup-2 | zfs receive -u backup/data

# Estimate stream size without sending data
zfs send -nPv -i tank/data@backup-1 tank/data@backup-2

# Remote backup; receiving account needs ZFS receive permissions
zfs send tank/data@backup-2 | ssh backup-host zfs receive -u backup/remote-data

# Replicate a dataset tree, properties, and snapshots through this snapshot
zfs snapshot -r tank/projects@backup-1
zfs send -R tank/projects@backup-1 | zfs receive -u backup/projects
```

Keep the backup destination unchanged between incremental receives. `-i` sends the difference between two snapshots; `-I` also includes intermediate snapshots. Do not add `zfs receive -F` casually: it can roll back destination changes and, for replication streams, destroy destination snapshots and filesystems absent from the source. In Bash backup scripts, use `set -o pipefail` so sender failures are not masked by the receiver's exit status.

## Maintenance and Disk Replacement

```bash
zpool scrub tank                        # Start checksum verification/repair
zpool status tank                       # Monitor scrub or resilver progress
zpool scrub -s tank                     # Stop a running scrub

# Replace a failed disk with one of sufficient capacity
zpool replace tank /dev/disk/by-id/OLD_DISK /dev/disk/by-id/NEW_DISK
zpool status tank                       # Wait for resilver to complete

# Temporarily take a disk offline, then return it
zpool offline tank /dev/disk/by-id/DISK_A
zpool online tank /dev/disk/by-id/DISK_A

# Clear recorded device errors AFTER diagnosing and fixing their cause
zpool clear tank
```

Check remaining redundancy before offlining a device. Schedule periodic scrubs and monitor their results. `zpool clear` does not repair corrupted files; permanent data errors may require restoration from backup.

## Expand Storage

```bash
# Add a second mirrored vdev: permanently changes the pool layout
zpool add -n tank mirror /dev/disk/by-id/DISK_C /dev/disk/by-id/DISK_D
zpool add tank mirror /dev/disk/by-id/DISK_C /dev/disk/by-id/DISK_D

# Turn an existing single-disk vdev into a mirror
zpool attach tank /dev/disk/by-id/DISK_A /dev/disk/by-id/DISK_B

# Expand a device after its underlying storage has grown
zpool online -e tank /dev/disk/by-id/DISK_A
```

`add` adds a top-level vdev; `attach` adds a mirror member. Adding a single unmirrored data disk to a redundant pool makes that disk a single point of failure. Do not assume an added vdev can be removed. When replacing disks with larger ones, all members of a mirror or RAIDZ vdev must grow before its extra capacity becomes available. RAIDZ expansion support depends on the OpenZFS version and enabled features.

## Export and Import

```bash
zpool export tank                       # Unmount and release pool for transfer
zpool import                           # List pools available to import
zpool import tank                      # Import an exported pool
zpool import -d /dev/disk/by-id tank     # Search a specific device directory
zpool import -N -o readonly=on tank      # Inspect without mounting or writing
zpool import -R /mnt/recovery tank      # Alternative import with temporary mount root
```

Import examples are alternatives. Never import a pool for writing on two hosts at once. Do not force an import until you have confirmed the previous host no longer has access. Pool feature upgrades can prevent older systems from importing the pool.

## Destructive Operations

```bash
# Preview recursive dataset deletion
zfs destroy -nvr tank/old-data

# DESTRUCTIVE: delete dataset, children, and their snapshots
zfs destroy -r tank/old-data

# DESTRUCTIVE: destroy the entire pool
zpool destroy tank
```

Verify names, backups, and dependencies before deletion. Lowercase `-r` recursively deletes descendants; uppercase `-R` can also destroy dependent clones outside that dataset tree.

## Reference

- [OpenZFS documentation](https://openzfs.github.io/openzfs-docs/)
- [OpenZFS manual pages](https://openzfs.github.io/openzfs-docs/man/index.html)
- Local manuals: `man zpool`, `man zfs`, `man zfsprops`, `man zpoolprops`
